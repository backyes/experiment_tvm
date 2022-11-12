import argparse
import json
import copy
from datetime import datetime, timezone


parser = argparse.ArgumentParser(description='Process log file.')

parser.add_argument('-f', type=str, help='log file')
parser.add_argument('--splitsec', type=float, default=0, help='split at seconds since start ')
parser.add_argument('--tfpf', type=str, default=None, help='TF trace  file')
parser.add_argument("--its", action='store_true', default=False, help="use iine number as timestamp")

args = parser.parse_args()

print("args: ", args)

def load_tf_trace(tracef, time_delta):
    tfpf=None
    if tracef is None:
        return tfpf
    with open(tracef) as f:
        tfpf=json.load(f) 
    for te in tfpf['traceEvents']:
        if 'ts' in te.keys():
            te['ts'] = time_delta + te['ts']
    return tfpf

def find_start_token(l):
        ltoks = l.split()
        starttok = -1
        for i,tok in enumerate(ltoks):
            if tok.startswith ('tensorflow/'):
                starttok = i
                break
            else:
                pass
        return starttok, ltoks


def runlength_encoding( lines):
    encoded = []
    runtok = ""
    runlength = 0
    for l in lines:

        starttok,ltoks = find_start_token(l)
        if starttok == -1: continue

        if ltoks[starttok] == runtok:
            runlength +=1 
        else:
            encoded.append( (runtok, runlength))
            runtok = ltoks[starttok]
            runlength = 1

    return encoded[1:] # removing the first dummy line 


def load_log( lines):
    encoded = []
    runtok = ""
    runlength = 0
    startts = 0
    splitts = -1
    splitline = -1
    for idx, l in enumerate(lines):

        starttok,ltoks = find_start_token(l)
        if starttok == -1: continue

        naive = datetime.strptime(" ".join(ltoks[:2]), "%Y-%m-%d %H:%M:%S.%f:")
        ts = naive.timestamp()*1.0e6
        if startts == 0:
            startts = ts

        if ltoks[starttok] == runtok:
            runlength +=1 
        else:
            runtok = ltoks[starttok]
            runlength = 1

        encoded.append( (ltoks[starttok], runlength, ts, idx, " ".join(ltoks[:2]+ltoks[4:])  ) )

        if splitts == -1  and (ts - startts ) >=  (args.splitsec*1e6) :
            splitts = ts
            splitline = len(encoded) -1 # index starts at 0

    return encoded[1:], startts, splitts, splitline # removing the first dummy line 

def runlength_encoding( lines):
    encoded = []
    runtok = ""
    runlength = 0
    for l in lines:

        starttok,ltoks = find_start_token(l)
        if starttok == -1: continue

        if ltoks[starttok] == runtok:
            runlength +=1 
        else:
            encoded.append( (runtok, runlength))
            runtok = ltoks[starttok]
            runlength = 1

    return encoded[1:] # removing the first dummy line 

def unique_count (lines):
    call_dict = {}
    for l in lines:

        starttok,ltoks = find_start_token(l)
        if starttok == -1: continue

        if ltoks[starttok]  not in call_dict.keys():
            call_dict[ltoks[starttok]] = 1
        else:
            call_dict[ltoks[starttok]] += 1

    return call_dict

def create_log_trace (tsel, marker=None):
    tracedict = {"traceEvents":[],
                 "meta_user": "sg",
                }

    multiplier = 1000
    its = 0
    tsdbegin = None
    tsdend   = None

    for tse in tsel:
        fpath,linenum = tse[0].split(":")
        if "optimization of a group" in tse[-1] or "Running optimization " in tse[-1]:
            ph = "i"
            s = "g"
        else:
            ph = "X"
            s = None

        if not args.its:
            ts = tse[2]
        else:
            ts = its*multiplier
        fpathl = fpath.split("/") 

        if len (fpathl) >3 :
            pid = "/".join(fpathl[:3])
            pid += " "*30
            #tid = fpath
            tid = "/".join(fpathl[3:])
        else:
            pid = fpath

        linerest = " ".join([str(v) for v in tse[3:]])
        linerest_max = 100
        linerest = linerest[:linerest_max] if len (linerest) > linerest_max else linerest
        tsd = {"pid": pid,
               "tid": tid, #linenum[:-1],
               "ts": ts,
               #"dur":tse[1]*multiplier,
               "dur":multiplier,
               "ph":ph,
               "s":s,
               "name": tse[0][:-1],
               "args": {#"linenum":linenum[:-1],
                        #"seqid":its,
						"runlength": tse[1],
                        "linerest":linerest},
              }
        #ts += tse[1] 
        its += 1 
        tracedict["traceEvents"].append(tsd)
        tsdend = tsd
        if tsdbegin is None:
            tsdbegin = tsd

    if marker == "end":
        tsdend = copy.deepcopy(tsdend)
        tsdend["ph"] = "i"
        tsdend["s"] = "g"
        tsdend["name"] = "End of log"
        tsdend["args"] = {}
        tracedict["traceEvents"].append(tsdend)
    elif marker == "begin":
        tsdbegin = copy.deepcopy(tsdbegin)
        tsdbegin["ph"] = "i"
        tsdbegin["s"] = "g"
        tsdbegin["name"] = "Begining of log"
        tsdbegin["args"] = {}
        tracedict["traceEvents"].append(tsdbegin)


    return tracedict


with open(args.f) as f:
    lines = f.readlines()

call_dict = unique_count (lines)
print ( "======unique calls: ", len(call_dict.keys()))
for k in call_dict.keys():
    print (k, call_dict[k])

'''
runlength  = runlength_encoding( lines)
print ("=======runlength calls: ", len(runlength))
for l in runlength:
    print ( l )

print ("summary: ", len(call_dict.keys()) , len(runlength))

jsonf = args.f+".runlength.json"
crate_trace(runlength, jsonf)
'''

logfullline,startts, splitts,splitline = load_log(lines)
tft = load_tf_trace(args.tfpf, startts)
for si in [(0,splitline, "end"), (splitline+1,len(logfullline), "begin")]:
    if si[0] == si[1]:
        continue
    splitlog = logfullline[si[0]:si[1]]
    logtrace = create_log_trace(splitlog, si[2])
    if tft is not None:
        tft['traceEvents'] = tft['traceEvents']+logtrace['traceEvents']
    else:
        tft = logtrace['traceEvents']
    jsonf = args.f+f"lines_{si[0]}_{si[1]}.merged.json"
    print("writting to file ", jsonf)
    with open ( jsonf, "w") as fp:
        json.dump(tft,fp)

print ("done")

