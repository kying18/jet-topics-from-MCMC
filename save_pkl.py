import pickle as pkl 
import numpy as np
import csv 

def format_data(input_filename, lines=(1,2)):
    data = []
    with open(input_filename, 'rt') as f:
        all_lines = f.readlines()
        hist1 = all_lines[lines[0]].split(",")[1:-2]
        hist2 = all_lines[lines[1]].split(",")[1:-2]
        label1 = all_lines[lines[0]].split(",")[0]
        label2 = all_lines[lines[1]].split(",")[0]
        print(len(hist1), hist1)

    for hist in [hist1, hist2]:
        hist = np.array(list(map(lambda x: int(x), hist)))
        print(hist)
        data.append(format_hist(hist))

    return data, label1, label2
    

def format_hist(hist):
    # hist is list of integers rep histogram bins
    tot_n = sum(hist)
    normalized_hist = 1/tot_n * hist
    bin_error = np.zeros(len(hist)) # todo: figure this out

    return [[normalized_hist, bin_error, hist], tot_n]

if __name__ == '__main__':
    #pp150_1 pbpb150_0_10_1_wide pbpb150_0_10_1 pp150_1_zjet
    # ordering of lines is counterintuitive
    pairs = [(0, 3), (0, 2), (1, 2)] # pp and ppZJet, pp and PbPb, PbPb and PbPb wide
    for pair in pairs:
        data, label1, label2 = format_data("./data/150_1_withpbpb_N10.csv", pair)
        with open(f"inputs/150_1_withpbpb_N10_{label1}_{label2}.pickle", "wb+") as f:
            pkl.dump(data, f)