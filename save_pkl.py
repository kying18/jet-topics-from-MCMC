import pickle as pkl 
import numpy as np
import csv 

def format_data(input_filename, labels):
    data = []
    with open(input_filename, 'rt') as f:
        all_lines = f.readlines()
        for i in range(len(all_lines)):
            split_vals = all_lines[i].split(",")
            if split_vals[0] == labels[0]:
                hist1 = split_vals[1:-2]
            elif split_vals[0] == labels[0] + "_error":
                hist1_error = split_vals[1:-2]
            elif split_vals[0] == labels[1]:
                hist2 = split_vals[1:-2]
            elif split_vals[0] == labels[1] + "_error":
                hist2_error = split_vals[1:-2]

    for hist, hist_error in [(hist1, hist1_error), (hist2, hist2_error)]:
        hist = np.array(list(map(lambda x: int(x), hist)))
        hist_error = np.array(list(map(lambda x: float(x), hist_error)))
        data.append(format_hist(hist, hist_error))

    return data, labels[0], labels[1]
    

def format_hist(hist, hist_error):
    # hist is list of integers rep histogram bins
    tot_n = sum(hist)
    normalized_hist = 1/tot_n * hist
    normalized_bin_error = 1/tot_n * hist_error

    return [[normalized_hist, normalized_bin_error, hist], tot_n]

if __name__ == '__main__':
    #pp150_1 pbpb150_0_10_1_wide pbpb150_0_10_1 pp150_1_zjet
    # ordering of lines is counterintuitive
    pairs = [('pp150_1', 'pp150_1_zjet'), ('pp150_1', 'pbpb150_0_10_1'), ('pbpb150_0_10_1', 'pbpb150_0_10_1_wide')]
    for pair in pairs:
        data, label1, label2 = format_data("./data/150_1_histograms.csv", pair)
        with open(f"inputs/150_1_histograms_{label1}_{label2}.pickle", "wb+") as f:
            pkl.dump(data, f)