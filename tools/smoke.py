"""Exercise packaged CLI and C ABI, with optional pinned processor/ORT parity."""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import wave

ROOT=Path(__file__).resolve().parents[1]

def smoke(folder, reference=False, synth=True):
    import numpy as np
    exe=folder/('grain-elevator.exe' if os.name=='nt' else 'grain-elevator')
    library=folder/('grain_elevator.dll' if os.name=='nt' else 'libgrain_elevator.so')
    env=os.environ.copy()
    for key in list(env):
        if key.startswith(('GRAIN_','SOUNDPRINT_','SP_','HF_','PYTHON')): env.pop(key,None)
    env['PATH']=str(Path(os.environ['SystemRoot'])/'System32') if os.name=='nt' else '/usr/bin:/bin'
    info=json.loads(subprocess.check_output([str(exe),'info'],env=env,text=True))
    rate=48000 if info['frontend']=='clap-v1' else 16000
    samples=np.array([0.3*math.sin(2*math.pi*220*i/rate) for i in range(rate)],dtype=np.float32)
    (ROOT/'test-results').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'test-results') as scratch:
        scratch=Path(scratch)
        audio=scratch/'synth-音.wav'
        with wave.open(str(audio),'wb') as output:
            output.setnchannels(1);output.setsampwidth(2);output.setframerate(rate)
            output.writeframes(b''.join(struct.pack('<h',round(float(v)*32767)) for v in samples))
        before=hashlib.sha256(audio.read_bytes()).hexdigest()
        output=scratch/'vector.f32'
        subprocess.run([str(exe),'embed',str(audio),'--output',str(output)],env=env,check=True)
        values=np.fromfile(output,dtype='<f4')
        metadata=json.loads(output.with_suffix('.metadata.json').read_text())
        assert len(values)==metadata['count'] and np.isfinite(values).all()
        assert hashlib.sha256(audio.read_bytes()).hexdigest()==before
        if info['output']=='embedding':
            assert len(values)==512 and abs(float(np.linalg.norm(values))-1)<1e-5
        else:
            assert metadata['shape'][-1]==1536 and len(metadata['mask'])==metadata['shape'][1]
        # Test the shared library with weights embedded in the DLL/.so itself.
        os.environ['GRAIN_ORT_LIB']=str(folder/('onnxruntime.dll' if os.name=='nt' else 'libonnxruntime.so'))
        lib=C.CDLL(str(library))
        lib.ge_open.argtypes=[C.c_char_p,C.c_size_t];lib.ge_open.restype=C.c_void_p
        lib.ge_embed_pcm.argtypes=[C.c_void_p,C.POINTER(C.c_float),C.c_size_t,C.c_uint32];lib.ge_embed_pcm.restype=C.c_void_p
        lib.ge_embed_file.argtypes=[C.c_void_p,C.c_char_p];lib.ge_embed_file.restype=C.c_void_p
        lib.ge_embedding_data.argtypes=[C.c_void_p,C.POINTER(C.c_size_t)];lib.ge_embedding_data.restype=C.POINTER(C.c_float)
        lib.ge_embedding_shape.argtypes=[C.c_void_p,C.POINTER(C.c_size_t)];lib.ge_embedding_shape.restype=C.POINTER(C.c_size_t)
        lib.ge_embedding_mask.argtypes=[C.c_void_p,C.POINTER(C.c_size_t)];lib.ge_embedding_mask.restype=C.POINTER(C.c_bool)
        lib.ge_embedding_free.argtypes=[C.c_void_p];lib.ge_close.argtypes=[C.c_void_p]
        lib.ge_last_error.argtypes=[C.c_char_p,C.c_size_t];lib.ge_last_error.restype=C.c_size_t
        def error():
            buffer=C.create_string_buffer(4096);lib.ge_last_error(buffer,len(buffer));return buffer.value.decode()
        encoder=lib.ge_open(str(library).encode(),1)
        assert encoder,error()
        try:
            result=lib.ge_embed_file(encoder,str(audio).encode())
            assert result,error()
            try:
                count=C.c_size_t();pointer=lib.ge_embedding_data(result,C.byref(count))
                from_file=np.ctypeslib.as_array(pointer,shape=(count.value,)).copy()
                assert np.array_equal(values,from_file),'CLI and ABI disagree'
                rank=C.c_size_t();shape=lib.ge_embedding_shape(result,C.byref(rank))
                assert list(shape[:rank.value])==metadata['shape']
                mask_count=C.c_size_t();mask=lib.ge_embedding_mask(result,C.byref(mask_count))
                assert list(mask[:mask_count.value])==metadata['mask']
            finally:lib.ge_embedding_free(result)
            vectors=[]
            for _ in range(2):
                result=lib.ge_embed_pcm(encoder,samples.ctypes.data_as(C.POINTER(C.c_float)),len(samples),rate)
                assert result,error()
                try:
                    count=C.c_size_t();pointer=lib.ge_embedding_data(result,C.byref(count))
                    vectors.append(np.ctypeslib.as_array(pointer,shape=(count.value,)).copy())
                finally:lib.ge_embedding_free(result)
            assert np.array_equal(*vectors),'repeatability failed'
            assert not lib.ge_embed_pcm(encoder,None,0,rate)
            assert 'invalid PCM' in error()
            parity=compare_reference(info,folder,samples,vectors[0]) if reference else None
        finally:lib.ge_close(encoder)
        if synth:
            report=ROOT/'test-results'/(info['key']+'-synth.json')
            subprocess.run([str(exe),'synth-test','--report',str(report)],env=env,check=True)
    summary={'model':info['key'],'revision':info['revision'],'cli':True,'c_abi':True,'repeatability':True,'source_unchanged':True,'reference':parity}
    (ROOT/'test-results'/(info['key']+'-smoke.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print('PASS native CLI/C ABI: '+info['key'],flush=True)
    return parity is None or parity['passed']

def compare_reference(info,folder,samples,actual):
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoProcessor
    from build import fetch
    processor_repo='laion/clap-htsat-unfused' if info['frontend']=='clap-v1' else 'google/embeddinggemma-2'
    revision='8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a' if info['frontend']=='clap-v1' else '914f7f89142e33e77833254d9c9b90c3cef7303b'
    processor=AutoProcessor.from_pretrained(processor_repo,revision=revision)
    inputs=processor(audio=[samples],sampling_rate=48000 if info['frontend']=='clap-v1' else 16000,return_tensors='np',**({'padding':'repeatpad'} if info['frontend']=='clap-v1' else {}))
    graph=fetch(info['repo'],info['revision'],info['artifact'],ROOT/'.hf-cache',False)
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    external_buffers=[]
    for part in info['parts'][1:]:
        path=fetch(info['repo'],info['revision'],part['name'],ROOT/'.hf-cache',False)
        buffer=path.read_bytes()
        external_buffers.append(buffer)
        options.add_external_initializers_from_files_in_memory([part['name']],[buffer],[len(buffer)])
    session=ort.InferenceSession(str(graph),options,providers=['CPUExecutionProvider'])
    feed={i.name:inputs[i.name].astype({'tensor(float)':np.float32,'tensor(int64)':np.int64,'tensor(bool)':bool}[i.type]) for i in session.get_inputs()}
    expected=session.run(None,feed)[0].reshape(-1)
    error=float(np.max(np.abs(expected-actual)))
    cosine=float(np.dot(expected.astype(np.float64),actual.astype(np.float64))/(np.linalg.norm(expected.astype(np.float64))*np.linalg.norm(actual.astype(np.float64))))
    # Quantized MatMul rounding can amplify frontend differences even when FP32
    # agrees within 1e-7. Keep the cosine gate equally strict for every variant.
    max_abs=5e-4 if info['quantization']=='fp32' else 5e-3
    passed=error<max_abs and cosine>0.9999
    print(f'Pinned processor/ORT parity: {"PASS" if passed else "FAIL"}, max_abs={error:.3g}, cosine={cosine:.9f}',flush=True)
    return {'max_abs_error':error,'cosine_similarity':cosine,'max_abs_limit':max_abs,'minimum_cosine':0.9999,'passed':passed}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('package',nargs='?',type=Path)
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--reference',action='store_true')
    parser.add_argument('--no-synth',action='store_true')
    args=parser.parse_args()
    folders=sorted(p for p in (ROOT/'dist').iterdir() if p.is_dir()) if args.all else [args.package]
    if not folders or any(p is None for p in folders):parser.error('provide a package or --all')
    # One Python process per package keeps the process-global ORT initializer isolated.
    if args.all:
        failed=False
        for folder in folders:
            result=subprocess.run([os.sys.executable,__file__,str(folder)]+(['--reference'] if args.reference else [])+(['--no-synth'] if args.no_synth else []))
            failed=failed or result.returncode!=0
        raise SystemExit(1 if failed else 0)
    else:raise SystemExit(0 if smoke(folders[0].resolve(),args.reference,not args.no_synth) else 1)
