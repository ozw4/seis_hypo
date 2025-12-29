# %%
from pathlib import Path

from nptdms import TdmsFile

# ====== EDIT HERE ======
tdms_path = Path(
	'/home/dcuser/daseventnet/data/silixa/raw_tdms/FORGE_DFIT_UTC_20220417_105956.202.tdms'
)
# =======================

tdms = TdmsFile.read(tdms_path)

print('=== TDMS ===')
print('path:', tdms_path)
print()

print('=== File properties ===')
for k, v in tdms.properties.items():
	print(f'- {k}: {v}')
print()

group_names = [g.name for g in tdms.groups()]
print('=== Groups ===')
print('count:', len(group_names))
for gname in group_names:
	print('-', gname)
print()

for g in tdms.groups():
	print(f'=== Group: {g.name} ===')
	if g.properties:
		print('properties:')
		for k, v in g.properties.items():
			print(f'  - {k}: {v}')
	else:
		print('properties: (none)')
	print('channels:', len(g.channels()))
	for ch in g.channels():
		print(f'  - {ch.name} | dtype={ch.data_type} | len={len(ch)}')
		if ch.properties:
			for k, v in ch.properties.items():
				print(f'      {k}: {v}')
	print()
# %%
