"""Small downloader/payload contract tests, without model downloads."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
import build

class BuildContracts(unittest.TestCase):
    def test_catalog_registers_nine_variants(self):
        models=build.catalog(build.ROOT/'models.json')
        self.assertEqual(len(models),9)
        self.assertEqual(sum(m['output']=='raw-tower' for m in models.values()),3)
        self.assertTrue(all(len(m['revision'])==40 for m in models.values()))

    def test_offline_url_uses_pin(self):
        models=build.catalog(build.ROOT/'models.json')
        key,spec=build.select_model('https://huggingface.co/bdrgon/clap-htsat-unfused-audio-embedder-onnx/resolve/main/audio_model_int8.onnx',models,True)
        self.assertEqual(key,'clap-int8')
        self.assertEqual(spec,models[key])
        with self.assertRaises(ValueError):build.select_model('https://other.example/model.onnx',models,True)
        with self.assertRaises(ValueError):build.select_model('https://huggingface.co/other/repo/resolve/main/file.onnx',models,True)

    def test_payload_offsets_and_no_source_mutation(self):
        models=build.catalog(build.ROOT/'models.json')
        scratch=build.ROOT/'.build'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            root=Path(temp)
            raw=root/'raw';raw.write_bytes(b'code')
            graph=root/'model.onnx';graph.write_bytes(b'graph')
            out=root/'out'
            digest=hashlib.sha256(b'graph').hexdigest()
            metadata=build.append_payload(raw,out,'clap-fp32',models['clap-fp32'],[('audio_model.onnx',graph,digest)])
            self.assertEqual(raw.read_bytes(),b'code')
            self.assertEqual(graph.read_bytes(),b'graph')
            data=out.read_bytes()
            self.assertEqual(data[-16:],build.MAGIC)
            n=struct.unpack('<Q',data[-24:-16])[0]
            self.assertEqual(json.loads(data[-24-n:-24]),metadata)
            part=metadata['parts'][0]
            self.assertEqual(data[part['offset']:part['offset']+part['length']],b'graph')

if __name__=='__main__':unittest.main()
