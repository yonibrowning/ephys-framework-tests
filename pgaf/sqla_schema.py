from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Float, Boolean

Base = declarative_base()

class Session(Base):
    __tablename__ = 'session'

    id = Column(Integer, primary_key=True)
    published_at = Column(String)#DateTime)
    specimen_id = Column(Integer)
    session_type = Column(String)
    age_in_days = Column(Integer)
    sex = Column(String)
    full_genotype = Column(String)
    unit_count = Column(Integer)
    channel_count = Column(Integer)
    probe_count = Column(Integer)

    def __repr__(self):
        return f'{self.specimen_id} {self.sex} {self.full_genotype} u={self.unit_count} cc={self.channel_count} pc={self.probe_count}'

class Structure(Base):
    __tablename__ = 'structure'
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    shortname = Column(String)

class SessionStructures(Base):
    __tablename__ = 'session_structures'
    
    id = Column(Integer, primary_key=True)
    
    structure_id = Column(Integer, ForeignKey('structure.id'))
    session_id = Column(Integer, ForeignKey('session.id'))

class ProbeStructures(Base):
    __tablename__ = 'probe_structures'
    
    id = Column(Integer, primary_key=True)
    
    structure_id = Column(Integer, ForeignKey('structure.id'))
    probe_id = Column(Integer, ForeignKey('probe.id'))

class Probe(Base):
    __tablename__ = 'probe'

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('session.id'))
    lfp_sampling_rate = Column(Float)
    sampling_rate = Column(Float)
    name = Column(String)
    phase = Column(String)
    has_lfp_data = Column(Boolean)
    unit_count = Column(Integer)
    channel_count = Column(Integer)    
    
                   
                
            
    
