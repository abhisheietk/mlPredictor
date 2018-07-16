import pandas as pd
import datetime
import numpy as np
import geopy.distance
import glob, os
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import clear_output, HTML
    
def normalize(df):    
    return (df - df.min()) / (df.max() - df.min())


def denormalize(df,norm_data):    
    return (norm_data * (df.max() - df.min())) + df.min()

def parseSlot(df):
    data = np.array(df[['counts_norm', 'bgCounts_norm']].values)
    return np.ndarray.flatten(data)

def getDate(timestamp):
    return datetime.datetime.fromtimestamp(int(timestamp))

def calTimeSpan(df):
    stop = getDate(dict(df.iloc[1])['datetime'])
    start = getDate(dict(df.iloc[-1])['datetime'])
    return start, stop

def getSweepSlots(start, stop, windowInHours = 5 * 24, shiftRateInHours = 1):
    delta = datetime.timedelta(hours=windowInHours)
    shift = datetime.timedelta(hours=shiftRateInHours)
    spanDelta = stop - start
    slots = int(spanDelta/shift)
    SweepSlots = []
    for slot in range(slots):
        SweepSlots.append([start + slot*shift, start + slot*shift + delta])
    return SweepSlots

def selectSweepData(df, slots):
    dSlots = []
    for slot in slots:
        start = slot[0].timestamp()
        stop  = slot[1].timestamp()
        #print(start, stop)
        dSlots.append({'start': start, 'stop': stop, 'data': df.loc[(df['datetime'] > start) & (df['datetime'] < stop)]})
    return dSlots

def filterBadSweeps(dSlots, dataPerSlot):
    newDSlots = []
    for dslot in dSlots:
        if len(dslot['data']) == dataPerSlot:
            newDSlots.append({'start': dslot['start'], 'stop': dslot['stop'], 'data': parseSlot(dslot['data'])})
    return newDSlots

def selectEventInSlot(eventdf, dSlots, eventMaxDistInKM, site, eventWindowInHours):
    #print(eventdf)
    for dSlot in dSlots:
        start = dSlot['stop']
        stop  = dSlot['stop'] + eventWindowInHours * 60 * 60
        events = eventdf.loc[(eventdf['datetime'] > start) &
                           (eventdf['datetime'] < stop)]
        #print(len(selectNearby(events, eventMaxDistInKM, site)))
        dSlot['output'] = len(selectNearby(events, eventMaxDistInKM, site)) > 0
    return dSlots

def selectNearby(eventdf, eventMaxDistInKM, site):
    siteCoords = (site['lat'], site['lng'])
    newEventDf = []
    for index, row in eventdf.iterrows():
        event = dict(row)
        eventCoords = (event['Latitude'], event['Longitude'])
        distance = geopy.distance.vincenty(siteCoords, eventCoords).km
        if distance <= eventMaxDistInKM:
            newEventDf.append(event)
    return pd.DataFrame(newEventDf)

def toDateTime(timeString):
    return datetime.datetime.strptime(timeString, '%d/%m/%Y %H:%M:%S').timestamp()

def finalize(eventDSlots):
    finalData = []    
    for eventDSlot in eventDSlots:
        data = {}
        inputData = eventDSlot['data']
        for i,j in enumerate(inputData):
            data[i] = j
        data['out'] = eventDSlot['output']
        finalData.append(data)
    return pd.DataFrame(finalData)

def NomMerge(fileNames, outFile):
    frames = []
    for fileName in fileNames:
        df = pd.read_csv(fileName, index_col=None, header=0)
        df.dropna(inplace=True)
        df_out = df['out']
        df_nom = normalize(df.drop(['out'],axis=1))
        df_nom['out']=df_out
        frames.append(df_nom)
    frame = pd.concat(frames)
    frame.to_csv(outFile, index=False)
    pass

def merge(fileNames, outFile):
    frames = []
    for fileName in fileNames:
        df = pd.read_csv(fileName, index_col=None, header=0)
        df['out']=df['out']*1
        df = df.dropna(inplace=False)
        msk = np.random.rand(len(df)) < 1
        df = df[msk]      
        frames.append(df)
    frame = pd.concat(frames)
    frame.to_csv(outFile, index=False)
    pass


def animateGraph(fileName, withPrediction=False):
    df = pd.read_csv(fileName)
    eventDf = df['out']
    if withPrediction:
        predDf = df['pred']
        df= df.drop('pred', axis=1)
    df= df.drop('out', axis=1)
    dataList = df.values 
    length = len(dataList[0])
    dlen = len(dataList)-1
    #dlen = 100
    
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_ylim((-0.2, 1))
    for i in range(10):
        ax.axvline(x=192*i, color='blue')
    for i in range(10, 13):
        ax.axvline(x=192*i, color='red')

    x = range(length)
    y = dataList[0]
    line, = ax.plot(x, y)
    #text = ax.text(0, 0.3, "Event: True",  bbox=dict(facecolor='white', alpha=0.5))
    eventTxt = ax.text(192*10, 0.3, "")
    predTxt = ax.text(192*10, 0.2, "")

    def init():    
        line.set_data([], [])
        eventTxt = ax.text(192*10, 0.3, "")
        predTxt = ax.text(192*10, 0.2, "")
        return (line, eventTxt, predTxt, )

    def animate(i):
        clear_output(wait=True)
        print('{0}/{1}'.format(i, dlen))
        y = dataList[dlen-i]
        line.set_data(x, y)
        if eventDf[dlen-i]:
            eventTxt.set_text("Event: {0}".format('True'))
            eventTxt.set_bbox(dict(facecolor='red', alpha=0.5))
        else:
            eventTxt.set_text("Event: {0}".format('False'))
            eventTxt.set_bbox(dict(facecolor='blue', alpha=0.5))
        if withPrediction:
            if predDf[dlen-i]:
                predTxt.set_text("Prediction: {0}".format('True'))
                predTxt.set_bbox(dict(facecolor='red', alpha=0.5))
            else:
                predTxt.set_text("Prediction: {0}".format('False'))
                predTxt.set_bbox(dict(facecolor='blue', alpha=0.5))
        else:
            predTxt.set_text("Prediction: Disabled")
            predTxt.set_bbox(dict(facecolor='green', alpha=0.5))
        return (line, eventTxt, predTxt, )

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=dlen, interval=20, blit=True)
    anim.save(fileName + '.mp4')
    #HTML(anim.to_html5_video())