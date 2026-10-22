#!/usr/bin/env python
"""Build native portable packages with weights fetched exclusively from HF/cache.

Python and Hugging Face tooling are build dependencies only. Model payloads are
appended to both the executable and C shared library, not stored beside them.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import tarfile
from urllib.parse import unquote, urlparse
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MAGIC = b'GRAIN-ELEVATOR01'

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def catalog(path):
    data = json.loads(Path(path).read_text())
    if data.get('schema') != 1 or not data.get('models'):
        raise ValueError('models.json must have schema=1 and nonempty models')
    return data['models']

def select_model(selection, models, offline=False):
    if selection in models:
        return selection, copy.deepcopy(models[selection])
    url = urlparse(selection)
    if url.scheme != 'https' or url.hostname != 'huggingface.co':
        raise ValueError('model must be a catalog key or https://huggingface.co/.../resolve/REV/FILE URL')
    segments = unquote(url.path).strip('/').split('/')
    if len(segments) < 5 or segments[2] not in ('resolve', 'blob'):
        raise ValueError('expected Hugging Face artifact URL with resolve or blob revision')
    repo, revision, artifact = '/'.join(segments[:2]), segments[3], '/'.join(segments[4:])
    matches = [(key, spec) for key, spec in models.items() if spec['repo'] == repo and spec['artifact'] == artifact]
    if len(matches) != 1:
        raise ValueError('unknown artifact frontend; register its adapter/contract in models.json first')
    key, spec = matches[0]
    spec = copy.deepcopy(spec)
    if offline:
        if revision not in ('main', spec['revision']):
            raise ValueError('offline URL revision must match the catalog pin (main uses the pin)')
    else:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        spec['revision'] = info.sha
        by_name = {f.rfilename: f for f in info.siblings}
        spec['sha256'] = by_name[artifact].lfs.sha256
        spec['sidecars'] = {name: by_name[name].lfs.sha256 for name in spec.get('sidecars', {})}
    return key, spec

def fetch(repo, revision, filename, cache, offline, expected_sha=None):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    # Read an existing standard HF cache first; never read adjacent model repos.
    for directory in [None, cache]:
        try:
            return Path(hf_hub_download(repo, filename, revision=revision, cache_dir=directory, local_files_only=True))
        except (FileNotFoundError, OSError, LocalEntryNotFoundError):
            pass
    # A README-only Hub commit need not trigger another large download on
    # Windows caches that keep real files rather than blob symlinks.
    if expected_sha:
        from huggingface_hub.constants import HF_HUB_CACHE
        for directory in [Path(HF_HUB_CACHE), Path(cache)]:
            snapshots=directory/('models--'+repo.replace('/','--'))/'snapshots'
            if snapshots.is_dir():
                for snapshot in snapshots.iterdir():
                    candidate=snapshot/filename
                    if candidate.is_file() and sha256(candidate)==expected_sha:
                        return candidate
    if offline:
        raise RuntimeError(f'not cached: {repo}@{revision}/{filename}')
    return Path(hf_hub_download(repo, filename, revision=revision, cache_dir=cache))

def external_locations(message):
    import onnx
    from google.protobuf.descriptor import FieldDescriptor
    if isinstance(message, onnx.TensorProto):
        if message.data_location == onnx.TensorProto.EXTERNAL:
            yield next((e.value for e in message.external_data if e.key == 'location'), '')
        return
    for field, value in message.ListFields():
        if field.type == FieldDescriptor.TYPE_MESSAGE:
            for item in value if field.is_repeated else [value]:
                yield from external_locations(item)

def resolve_artifacts(spec, cache, offline):
    import onnx
    graph = fetch(spec['repo'], spec['revision'], spec['artifact'], cache, offline, spec['sha256'])
    if sha256(graph) != spec['sha256']:
        raise RuntimeError('graph checksum disagrees with selected revision/catalog')
    model = onnx.load(str(graph), load_external_data=False)
    expected_inputs = ['input_features'] if spec['frontend'] == 'clap-v1' else (
        ['input_features', 'input_features_mask'] if spec['output'] == 'raw-tower' else
        ['input_ids', 'attention_mask', 'input_features', 'input_features_mask'])
    expected_outputs = ['audio_features', 'audio_mask'] if spec['output'] == 'raw-tower' else ['embedding']
    if [i.name for i in model.graph.input] != expected_inputs or [o.name for o in model.graph.output] != expected_outputs:
        raise RuntimeError('ONNX input/output contract disagrees with models.json')
    locations = set(external_locations(model))
    del model
    parts = [(spec['artifact'], graph, spec['sha256'])]
    for name in sorted(locations):
        if not name or Path(name).name != name or name not in spec.get('sidecars', {}):
            raise RuntimeError(f'unregistered external weights {name!r}; repair/register the Hub artifact, do not silently omit it')
        path = fetch(spec['repo'], spec['revision'], name, cache, offline, spec['sidecars'][name])
        digest = sha256(path)
        if digest != spec['sidecars'][name]:
            raise RuntimeError(f'weights checksum mismatch: {name}')
        parts.append((name, path, digest))
    if locations != set(spec.get('sidecars', {})):
        raise RuntimeError('catalog sidecars disagree with graph references')
    return parts

def append_payload(source, destination, key, spec, parts):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    metadata = {name: spec[name] for name in ['repo', 'revision', 'artifact', 'frontend', 'output', 'dimension', 'quantization']}
    metadata.update(schema=1, key=key, parts=[])
    with destination.open('ab') as out:
        for name, path, digest in parts:
            offset = out.tell()
            with path.open('rb') as data:
                shutil.copyfileobj(data, out, 8 * 1024 * 1024)
            metadata['parts'].append(dict(name=name, offset=offset, length=path.stat().st_size, sha256=digest))
        encoded = json.dumps(metadata, separators=(',', ':')).encode()
        out.write(encoded)
        out.write(struct.pack('<Q', len(encoded)))
        out.write(MAGIC)
    return metadata

def cargo_environment():
    env = os.environ.copy()
    env.setdefault('CARGO_BUILD_JOBS', '1')
    sibling = ROOT.parent / '.toolchain'
    if not shutil.which('cargo') and (sibling / 'rust/bin/cargo.exe').is_file():
        env['PATH'] = str(sibling / 'rust/bin') + os.pathsep + str(sibling / 'mingw/mingw64/bin') + os.pathsep + env['PATH']
        env['CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER'] = str(sibling / 'mingw/mingw64/bin/x86_64-w64-mingw32-gcc.exe')
    if 'CARGO_HOME' not in env and (ROOT.parent / '.cargo-home/registry').is_dir():
        env['CARGO_HOME'] = str(ROOT.parent / '.cargo-home')
    linker=sibling/'mingw/mingw64/bin/x86_64-w64-mingw32-gcc.exe'
    if linker.is_file():
        env.setdefault('CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER',str(linker))
    return env

def cargo_path(env):
    path = shutil.which('cargo', path=env['PATH'])
    if not path:
        raise RuntimeError('Install Rust/Cargo or put it on PATH')
    return path

def copy_runtime(folder, spec, vc_redist, allow_system_crt):
    import onnxruntime
    capi = Path(onnxruntime.__file__).parent / 'capi'
    name = spec['runtime']
    matches = sorted(capi.glob(name + '*'))
    runtime = next((p for p in matches if p.is_file()), None)
    if runtime is None:
        raise RuntimeError(f'{name} unavailable in this host Python; build on the target OS')
    shutil.copy2(runtime, folder / name)
    shared = capi / ('onnxruntime_providers_shared.dll' if spec['host'] == 'Windows' else 'libonnxruntime_providers_shared.so')
    if shared.is_file():
        shutil.copy2(shared, folder / shared.name)
    if spec['host'] == 'Windows':
        if vc_redist:
            crt = Path(vc_redist)
            for name in ['vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll', 'msvcp140_1.dll']:
                if not (crt / name).is_file():
                    raise RuntimeError(f'VC redistributable missing {name}')
            for dll in crt.glob('*.dll'):
                shutil.copy2(dll, folder / dll.name)
            license_folder=folder/'licenses'/'vc-runtime'
            license_folder.mkdir(parents=True,exist_ok=True)
            for document in crt.glob('*.txt'):
                shutil.copy2(document,license_folder/document.name)
        elif not allow_system_crt:
            raise RuntimeError('portable Windows package needs --vc-redist LICENSED_CRT_DIR; --allow-system-crt is development-only')

def notices(folder, env, parts, spec, cache, offline):
    dest = folder / 'licenses'
    dest.mkdir(exist_ok=True)
    for name in ['LICENSE', 'NOTICE']:
        try:
            source = fetch(spec['repo'], spec['revision'], name, cache, offline)
        except Exception:
            if name == 'LICENSE':
                raise
        else:
            shutil.copy2(source, dest / ('model-' + name))
    import onnxruntime
    for source in Path(onnxruntime.__file__).parent.glob('*'):
        if source.is_file() and any(x in source.name.lower() for x in ['license','notice']):
            shutil.copy2(source, dest / ('onnxruntime-' + source.name))
    metadata_command=[cargo_path(env),'metadata','--locked','--format-version','1']
    if offline: metadata_command.append('--offline')
    data = json.loads(subprocess.check_output(metadata_command,cwd=ROOT,env=env))
    rows=[]
    for package in data['packages']:
        rows.append(f"{package['name']} {package['version']}: {package.get('license') or 'SEE LICENSE'}")
        source = Path(package['manifest_path']).parent
        for candidate in source.iterdir():
            if candidate.is_file() and candidate.name.upper().startswith(('LICENSE','COPYING','NOTICE')):
                target = dest / (package['name'] + '-' + package['version'])
                target.mkdir(exist_ok=True)
                shutil.copy2(candidate, target / candidate.name)
    (dest / 'dependencies.txt').write_text('\n'.join(rows)+'\n')
    shutil.copy2(ROOT / 'NOTICE', dest / 'grain-elevator-NOTICE')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='catalog key, all, or public HF artifact URL')
    parser.add_argument('--models', type=Path, default=ROOT/'models.json')
    platforms=json.loads((ROOT/'platforms.json').read_text())['platforms']
    parser.add_argument('--platform', choices=list(platforms), default='windows-x64' if platform.system()=='Windows' else 'linux-x64')
    parser.add_argument('--cache-dir', type=Path, default=ROOT/'.hf-cache')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--vc-redist', type=Path)
    parser.add_argument('--allow-system-crt', action='store_true')
    parser.add_argument('--no-archive', action='store_true')
    args=parser.parse_args()
    models=catalog(args.models)
    target=platforms[args.platform]
    if platform.system()!=target['host'] or platform.machine() not in target['machine']:
        raise RuntimeError('build on the selected native host; see the Linux/Windows build paths and CI matrix')
    env=cargo_environment()
    cmd=[cargo_path(env),'build','--release','--locked','--target',target['target']]
    if args.offline: cmd.append('--offline')
    subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    raw=ROOT/'target'/target['target']/'release'
    for selection in models if args.model=='all' else [args.model]:
        key,spec=select_model(selection,models,args.offline)
        print(f"Resolving {key}: {spec['repo']}@{spec['revision']}/{spec['artifact']}",flush=True)
        parts=resolve_artifacts(spec,args.cache_dir,args.offline)
        folder=ROOT/'dist'/f'{key}-{args.platform}'
        folder.mkdir(parents=True,exist_ok=True)
        metadata=append_payload(raw/target['exe'],folder/target['exe'],key,spec,parts)
        append_payload(raw/target['library'],folder/target['library'],key,spec,parts)
        copy_runtime(folder,target,args.vc_redist,args.allow_system_crt)
        notices(folder,env,parts,spec,args.cache_dir,args.offline)
        shutil.copy2(ROOT/'include/grain_elevator.h',folder/'grain_elevator.h')
        (folder/'model.json').write_text(json.dumps(metadata,indent=2)+'\n')
        (folder/'README.txt').write_text('Offline Grain Elevator package. Run grain-elevator --help.\nWeights are embedded in both the executable and shared library.\n'+('Development package: requires installed Visual C++ x64 runtime.\n' if args.allow_system_crt and target['host']=='Windows' else ''))
        files=sorted(p for p in folder.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
        (folder/'SHA256SUMS').write_text(''.join(f'{sha256(p)}  {p.relative_to(folder).as_posix()}\n' for p in files))
        if not args.no_archive:
            if target['archive']=='zip':
                with zipfile.ZipFile(folder.with_suffix('.zip'),'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as archive:
                    for file in sorted(folder.rglob('*')):
                        if file.is_file(): archive.write(file,file.relative_to(folder))
            else:
                with tarfile.open(str(folder)+'.tar.gz','w:gz') as archive: archive.add(folder,arcname=folder.name)
        print(f'Built {folder}',flush=True)

if __name__=='__main__':
    try: main()
    except Exception as error:
        print(f'Build failed: {error}',file=sys.stderr)
        sys.exit(1)
