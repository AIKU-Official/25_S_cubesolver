import pickle, json
from pathlib import Path
import pandas as pd
import numpy as np

# 로드
with open('results/cube2/results.pkl', 'rb') as f:
    b = pickle.load(f)

out_prefix = 'results/cube2/b'
Path('results/cube2').mkdir(parents=True, exist_ok=True)

# 2-1) 항상 피클 백업
with open(f'{out_prefix}.pkl', 'wb') as f:
    pickle.dump(b, f, protocol=pickle.HIGHEST_PROTOCOL)

# 2-2) CSV 시도 (DataFrame/Series/list[dict]/dict-of-sequences)
def try_save_csv(obj, path):
    try:
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(path, index=False); return True
        if isinstance(obj, pd.Series):
            obj.to_frame().to_csv(path, index=False); return True
        if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
            pd.DataFrame(obj).to_csv(path, index=False); return True
        if isinstance(obj, dict):
            vals = list(obj.values())
            if vals and all(hasattr(v, "__len__") for v in vals):
                lens = {len(v) for v in vals}
                if len(lens) == 1:
                    pd.DataFrame(obj).to_csv(path, index=False); return True
        if isinstance(obj, np.ndarray) and getattr(obj.dtype, "names", None):
            pd.DataFrame.from_records(obj).to_csv(path, index=False); return True
    except Exception as e:
        print("CSV 저장 건너뜀:", e)
    return False

csv_ok = try_save_csv(b, f'{out_prefix}.csv')

# 2-3) JSON 폴백
if not csv_ok:
    try:
        with open(f'{out_prefix}.json', 'w', encoding='utf-8') as f:
            json.dump(b, f, ensure_ascii=False, default=str, indent=2)
        print("CSV로 만들기 어려워 JSON으로 저장했습니다.")
    except TypeError:
        # 마지막 폴백: 텍스트로 repr 덤프
        Path(f'{out_prefix}.txt').write_text(repr(b), encoding='utf-8')
        print("JSON 직렬화 불가 → 텍스트(repr)로 저장했습니다.")