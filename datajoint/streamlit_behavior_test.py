import streamlit as st
import numpy as np
import pandas as pd
import datajoint as dj
from matplotlib import pyplot as plt

# Because streamlit doesn't seem to reload packages,
# This is just here while I debug some stuff

from importlib import reload
plt = reload(plt)

import yoni_behavior_schema
from yoni_behavior_schema import Session,Probe,Channel,Unit,SpikeTrain,TrialSpikeCount,Stimulus,GaborRF,StaticResponse,DriftingResponse

# Set up the title
st.title('DataJoint Visualization')

sessions = Session().fetch(format = 'frame')
session_option = st.sidebar.selectbox('Session ID', sessions.index)

# Brain Area selector
brain_area_list = np.unique((Channel()&(Probe()&['session_id = '+str(session_option)])).fetch('brain_structure'))

brain_area_option = st.sidebar.selectbox('Brain Area', brain_area_list)

# Select the unit, get the spike train
unit_option = st.sidebar.selectbox('Unit ID', (SpikeTrain()&(Unit()&(Channel()&['brain_structure = "'+
brain_area_option +'"']&(Probe() &['session_id = '+str(session_option)])))).fetch('unit_id'))
spike_ts = (SpikeTrain()&['unit_id = '+str(unit_option)]).fetch1('spike_ts')

# Make a ISI histogram
isi_container = st.container()

fig1 = plt.figure()
isi_container.subheader("ISI Distribution")
plt.hist(np.diff(spike_ts),np.arange(0,500)*.001)
plt.xlabel('Inter-Spike-Interval (sec)')
plt.ylabel('# of Spikes')
isi_container.pyplot(fig1)

# Set up container for receptive fields
rf_container = st.container()
rf_col,grating_col = rf_container.columns(2)
rf_col.subheader("Spatial Receptive Field")
grating_col.subheader("Grating Selectivity")


fig2 = plt.figure()
rfdata = (GaborRF()&['session_id = '+str(session_option)]&['unit_id = '+str(unit_option)]).fetch1()


plt.pcolor(rfdata['receptive_field'])
plt.xticks(.5+np.arange(len(rfdata['x'])),rfdata['x'])
plt.yticks(.5+np.arange(len(rfdata['y'])),rfdata['y'])
plt.clim(0,np.max(rfdata['receptive_field']))
cbar = plt.colorbar()
cbar.set_label('Rate (hz)')
plt.xlabel('X position')
plt.ylabel('Y position')
rf_col.pyplot(fig2)

fig3 = plt.figure()
sdata = (StaticResponse()&['session_id = '+str(session_option)]&['unit_id = '+str(unit_option)]).fetch1()
plt.pcolor(sdata['static_response'])
plt.xticks(.5+np.arange(len(sdata['orientation'])),sdata['orientation'])
plt.yticks(.5+np.arange(len(sdata['spatial_frequency'])),sdata['spatial_frequency'])
plt.clim(0,np.max(sdata['static_response']))
cbar = plt.colorbar()
cbar.set_label('Rate (hz)')
plt.xlabel('Orientation (deg)')
plt.ylabel('Spatial Freq')
grating_col.pyplot(fig3)

fig4 = plt.figure()
ddata = (DriftingResponse()&['session_id = '+str(session_option)]&['unit_id = '+str(unit_option)]).fetch1()
plt.pcolor(ddata['drifting_response'])
plt.xticks(.5+np.arange(len(ddata['orientation'])),ddata['orientation'])
plt.yticks(.5+np.arange(len(ddata['temporal_frequency'])),ddata['temporal_frequency'])
plt.clim(0,np.max(ddata['drifting_response']))
cbar = plt.colorbar()
cbar.set_label('Rate (hz)')
plt.xlabel('Orientation (deg)')
plt.ylabel('Temporal Freq')
grating_col.pyplot(fig4)










