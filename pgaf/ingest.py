from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

import sqla_schema as sch


manifest = '/allen/scratch/aindtemp/david.feng/epc/manifest.json'
cache = EcephysProjectCache(manifest=manifest)

sessions = cache.get_session_table()
cache.get_probes()
cache.get_channels()
cache.get_units()

for idx,row in sessions.head(3).iterrows():
    cache.get_session_data(idx)


engine = create_engine('sqlite://')
sch.Base.metadata.create_all(engine)

sessions.to_sql('session', engine, if_exists='replace')

with Session(engine) as sess:
    q = select(sch.Session).where(sch.Session.sex == 'M')
    for row in sess.execute(q):
        print(dict(row))
    

