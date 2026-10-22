"""Development-only parity fixtures from the pinned HF Gemma processor."""
from pathlib import Path
import json
import numpy as np
from transformers import AutoProcessor
root=Path(__file__).resolve().parents[1]
p=AutoProcessor.from_pretrained('google/embeddinggemma-2',revision='914f7f89142e33e77833254d9c9b90c3cef7303b')
cases=[]
for n in [1920,3280,16000]:
    samples=(0.3*np.sin(2*np.pi*220*np.arange(n)/16000)).astype(np.float32)
    inputs=p(audio=[samples],sampling_rate=16000,return_tensors='np')
    cases.append({'samples':samples.tolist(),'features':inputs['input_features'][0].reshape(-1).tolist(),'mask':inputs['input_features_mask'][0].tolist(),'ids':inputs['input_ids'][0].tolist()})
folder=root/'tests/fixtures'
folder.mkdir(exist_ok=True)
(folder/'gemma_frontend.json').write_text(json.dumps({'processor_revision':'914f7f89142e33e77833254d9c9b90c3cef7303b','transformers':'5.19.0','cases':cases},separators=(',',':'))+'\n')
print('Saved Gemma frontend parity fixtures')
