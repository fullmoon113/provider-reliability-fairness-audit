"""Extract the columns used by the Medicare analyses from the CMS 2022 "by Provider" file (MUP_PHY_R25_P07_V20_D22_Prov.csv).
    python src/make_cms2022_extract.py path/to/MUP_PHY_R25_P07_V20_D22_Prov.csv  cms2022_extract.csv.gz"""
import sys, pandas as pd
from flag_rate_common import EXTRACT_COLS
d = pd.read_csv(sys.argv[1], usecols=EXTRACT_COLS, low_memory=False)
d = d[d.Tot_Benes > 10]; d.to_csv(sys.argv[2], index=False, compression='gzip'); print(len(d), 'providers written to', sys.argv[2])
