import numpy as np
import matplotlib.pyplot as plt
import h5py
import plot_utility

REAL_LENGTH = 200
HDF_LENGTH = 20
SCALE = HDF_LENGTH/REAL_LENGTH
BINNR = 20
SHUFFLE_N = 100
STOP_THRESHOLD = 0.7
DIST = 0.1

remove_late_stops = False
account_for_no_stop_runs=True


def analyse(filename, first_stop=False, cummulative=False):
    bin_size = 5
    bins = np.arange(0, 200, bin_size)
    bin_centres = 0.5*(bins[1:]+bins[:-1])

    # specify mouse/mice and day/s to analyse
    days = ['Day' + str(int(x)) for x in np.arange(1,22.1)]
    mice = ['M' + str(int(x)) for x in np.arange(1,9.1)]
    n_days = len(days)

    # For each day and mouse, pull raw data, calculate first stops and store data
    for mcount,mouse in enumerate(mice):

        beaconed_stops_early = []
        non_beaconed_stops_early = []
        b_proportion_early = []
        nb_proportion_early = []

        beaconed_stops_late = []
        non_beaconed_stops_late = []
        b_proportion_late = []
        nb_proportion_late = []

        for dcount,day in enumerate(days):
            dcount+=1
            try:
                saraharray = readhdfdata(filename,day,mouse,'raw_data')
            except KeyError:
                print ('Error, no file')
                continue
            print('##...', mcount,day, '...##')

            # make array of trial number for each row in dataset
            trialarray = maketrialarray(saraharray) # write array of trial per row in datafile
            saraharray[:,9] = trialarray[:,0] # replace trial column in dataset *see README for why this is done*

            # split data by trial type
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0)
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)

            # get stops
            if first_stop:
                beaconed_stops, n_beaconed_trials_with_stops, n_beaconed_trials        = extractfirststops2(dailymouse_b)
                non_beaconed_stops, n_nonbeaconed_trials_with_stops, n_nonbeaconed_trials = extractfirststops2(dailymouse_nb)
                probe_stops, n_probe_trials_with_stops, n_probe_trials              = extractfirststops2(dailymouse_p)
            else:
                beaconed_stops, n_beaconed_trials_with_stops, n_beaconed_trials        = extractstops2(dailymouse_b)
                non_beaconed_stops, n_nonbeaconed_trials_with_stops, n_nonbeaconed_trials    = extractstops2(dailymouse_nb)
                probe_stops, n_probe_trials_with_stops, n_probe_trials           = extractstops2(dailymouse_p)

            beaconed_stop_histogram, _ = np.histogram(beaconed_stops, bins)
            beaconed_stop_histogram = beaconed_stop_histogram/n_beaconed_trials
            non_beaconed_stop_histogram, _ = np.histogram(non_beaconed_stops, bins)
            non_beaconed_stop_histogram = non_beaconed_stop_histogram/n_nonbeaconed_trials

            b_proportion_stopped = n_beaconed_trials_with_stops/n_beaconed_trials
            nb_proportion_stopped = n_nonbeaconed_trials_with_stops/n_nonbeaconed_trials

            if not account_for_no_stop_runs:
                b_proportion_stopped = 1   # comment these out to get the right
                nb_proportion_stopped = 1

            beaconed_stop_histogram = beaconed_stop_histogram/np.sum(beaconed_stop_histogram)
            non_beaconed_stop_histogram = non_beaconed_stop_histogram/np.sum(non_beaconed_stop_histogram)

            if cummulative:
                beaconed_stop_histogram = np.cumsum(beaconed_stop_histogram)
                non_beaconed_stop_histogram = np.cumsum(non_beaconed_stop_histogram)

            if dcount<5:
                beaconed_stops_early.append(beaconed_stop_histogram)
                non_beaconed_stops_early.append(non_beaconed_stop_histogram)
                b_proportion_early.append(b_proportion_stopped)
                nb_proportion_early.append(nb_proportion_stopped)
            elif n_days-dcount<=5:
                beaconed_stops_late.append(beaconed_stop_histogram)
                non_beaconed_stops_late.append(non_beaconed_stop_histogram)
                b_proportion_late.append(b_proportion_stopped)
                nb_proportion_late.append(nb_proportion_stopped)

        beaconed_stops_early = np.array(beaconed_stops_early)
        non_beaconed_stops_early = np.array(non_beaconed_stops_early)
        beaconed_stops_late = np.array(beaconed_stops_late)
        non_beaconed_stops_late = np.array(non_beaconed_stops_late)
        b_proportion_early = np.array(b_proportion_early)
        nb_proportion_early = np.array(nb_proportion_early)
        b_proportion_late = np.array(b_proportion_late)
        nb_proportion_late = np.array(nb_proportion_late)

        avg_beaconed_stops_early = np.nanmean(beaconed_stops_early, axis=0)
        avg_non_beaconed_stops_early = np.nanmean(non_beaconed_stops_early, axis=0)
        avg_beaconed_stops_late = np.nanmean(beaconed_stops_late, axis=0)
        avg_non_beaconed_stops_late = np.nanmean(non_beaconed_stops_late, axis=0)

        std_beaconed_stops_early = np.nanstd(beaconed_stops_early, axis=0)
        std_non_beaconed_stops_early = np.nanstd(non_beaconed_stops_early, axis=0)
        std_beaconed_stops_late = np.nanstd(beaconed_stops_late, axis=0)
        std_non_beaconed_stops_late = np.nanstd(non_beaconed_stops_late, axis=0)

        avg_b_proportion_early = np.nanmean(b_proportion_early)
        avg_nb_proportion_early = np.nanmean(nb_proportion_early)
        avg_b_proportion_late = np.nanmean(b_proportion_late)
        avg_nb_proportion_late = np.nanmean(nb_proportion_late)

        avg_beaconed_stops_early=avg_beaconed_stops_early*avg_b_proportion_early
        avg_non_beaconed_stops_early=avg_non_beaconed_stops_early*avg_nb_proportion_early
        avg_beaconed_stops_late= avg_beaconed_stops_late*avg_b_proportion_late
        avg_non_beaconed_stops_late=avg_non_beaconed_stops_late*avg_nb_proportion_late

        std_beaconed_stops_early=std_beaconed_stops_early*avg_b_proportion_early
        std_non_beaconed_stops_early=std_non_beaconed_stops_early*avg_nb_proportion_early
        std_beaconed_stops_late= std_beaconed_stops_late*avg_b_proportion_late
        std_non_beaconed_stops_late=std_non_beaconed_stops_late*avg_nb_proportion_late

        b_y_max = max(max(avg_beaconed_stops_early), max(avg_beaconed_stops_late))
        nb_y_max = max(max(avg_non_beaconed_stops_early), max(avg_non_beaconed_stops_late))

        # first stop histogram
        fig = plt.figure(figsize = (12,4))
        ax = fig.add_subplot(1,2,1) #stops per trial
        ax.set_title('Beaconed', fontsize=20, verticalalignment='bottom', style='italic')  # title
        ax.plot(bin_centres,avg_beaconed_stops_early,color = 'black',label = 'First 5 days', linewidth = 2) #plot becaoned trials
        ax.fill_between(bin_centres,avg_beaconed_stops_early-std_beaconed_stops_early,avg_beaconed_stops_early+std_beaconed_stops_early, facecolor = 'black', alpha = 0.3)
        ax.plot(bin_centres,avg_beaconed_stops_late,color = 'red',label = 'Last 5 days', linewidth = 2) #plot becaoned trials
        ax.fill_between(bin_centres,avg_beaconed_stops_late-std_beaconed_stops_late,avg_beaconed_stops_late+std_beaconed_stops_late, facecolor = 'red', alpha = 0.3)
        ax.set_xlim(0,200)
        ax.set_ylabel('Stop Probability', fontsize=16, labelpad = 18)
        plot_utility.style_vr_plot_offset(ax, max(avg_beaconed_stops_late))
        plot_utility.style_track_plot(ax, 200)
        ax.set_ylim(0, b_y_max+0.01)

        # non beaconed stop histogram
        ax = fig.add_subplot(1,2,2) #stops per trial
        ax.set_title('Non-Beaconed', fontsize=20, verticalalignment='bottom', style='italic')  # title
        ax.plot(bin_centres,avg_non_beaconed_stops_early,color = 'black', label = 'First 5 days', linewidth = 2) #plot becaoned trials
        ax.fill_between(bin_centres,avg_non_beaconed_stops_early-std_non_beaconed_stops_early,avg_non_beaconed_stops_early+std_non_beaconed_stops_early, facecolor = 'black', alpha = 0.3)
        ax.plot(bin_centres,avg_non_beaconed_stops_late,color = 'red', label = 'Last 5 days', linewidth = 2) #plot becaoned trials
        ax.fill_between(bin_centres,avg_non_beaconed_stops_late-std_non_beaconed_stops_late,avg_non_beaconed_stops_late+std_non_beaconed_stops_late, facecolor = 'red', alpha = 0.3)
        ax.set_xlim(0,200)
        #ax.set_xlabel("Track Position relative to goal (cm)", fontsize=16, labelpad = 18)
        plot_utility.style_vr_plot_offset(ax, max(avg_beaconed_stops_late))
        plot_utility.style_track_plot(ax, 200)
        ax.set_ylim(0, nb_y_max+0.01)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.6, left = 0.15, right = 0.82, top = 0.85)
        fig.text(0.5, 0.04, 'Track Position Relative to Goal (cm)', ha='center', fontsize=16)
        fig.text(0.05, 0.94, mouse, ha='center', fontsize=16)
        ax.legend(loc=(0.99, 0.5))
        plt.show()
        #fig.savefig(save_path,  dpi=200)
        #plt.close()


        mcount +=1

# -------------------------------------------------------------------------------------------------------------- #

# FUNCTION FOR READING HDF5 FILES
def readhdfdata(filename,day,mouse,dataset):
    fopen = h5py.File(filename, 'r')
    datasetopen  = fopen[day + '/' + mouse  + '/' + dataset]
    openarray = datasetopen[:,:]
    fopen.close()
    return openarray

# FUNCTION TO EXTRACT STOPS FROM DATASET
#input: array[:,13] (columns: for raw data, see README.py file)
#output: array[:,3] (columns: location, time, trialno, reward)
#function: extracts row of data if speed drops below 0.7 cm/s.
def extractstops(stops):
    moving = False
    data = []
    for row in stops:
        if(row[2]<=STOP_THRESHOLD and moving): # if speed is below threshold
            moving = False
            data.append([float(row[1])+0.2, float(row[0]), int(row[9]), int(row[4])]) # location, (beaconed/non-beaconed/probe), trialid, reward(YES/NO)

        if(row[2]>STOP_THRESHOLD and not moving):
            moving = True
    return np.array(data)

def extractfirststops2(stops):
    was_below = False
    data = []
    trials_with_stops = []
    trial_numbers = []
    for row in stops:
        if (int(row[9]) not in trial_numbers):
            trial_numbers.append((int(row[9])))

        if(row[2]<=STOP_THRESHOLD and not was_below): # if speed is below threshold
            if (int(row[9]) not in trials_with_stops):
                data.append((float(row[1])+0.2)*10)
                trials_with_stops.append(int(row[9]))
            was_below = True

        if(row[2]>STOP_THRESHOLD):
            was_below = False

    return np.array(data), len(trials_with_stops), len(trial_numbers)

def extractstops2(stops, first_stop=False):
    was_below = False
    data = []
    trials_with_stops = []
    trial_numbers = []
    for row in stops:
        if (int(row[9]) not in trial_numbers):
            trial_numbers.append((int(row[9])))

        if(row[2]<=STOP_THRESHOLD and not was_below): # if speed is below threshold
            data.append((float(row[1])+0.2)*10)
            was_below = True

            if (int(row[9]) not in trials_with_stops):
                trials_with_stops.append(int(row[9]))

        if(row[2]>STOP_THRESHOLD):
            was_below = False

    return np.array(data), len(trials_with_stops), len(trial_numbers)
#================================================================================================================================================================#


# MAKE LOCATION BINS
def makebinarray(tarray, bins):
    interval = 1
    binarray = np.zeros((tarray.shape[0], 1))
    for bcount,b in enumerate(bins): # Collect data for each bin
        binmin = tarray[:,1]>=b # lowest value in bin
        binmax = tarray[:,1]<b+interval # highest value in bin
        arraylogical = np.logical_and(binmin,binmax) #get all rows that satisfy being within bin
        binarray[arraylogical, 0] = b #assign each row in tarray its respective bin
    return binarray


# MAKE ARRAY OF TRIAL NUMBER FOR EVERY ROW IN DATA
#input: array[:,13] (columns: for raw data, see README.py file)
#output: array[:,0] (columns: trialno)
#function: remove stops that occur 1 cm after a stop
def maketrialarray(saraharray):
    trialarray = np.zeros((saraharray.shape[0],1)) # make empty array same row number as datafile
    trialnumber = 1
    trialarray[0] = 1 #because the first row is always in the first trial
    for rowcount, row in enumerate(saraharray[:-1, :]): # for every row in data
        if saraharray[rowcount + 1, 1]-saraharray[rowcount,1]<-10: # if current location - last location is less than -10
            trialnumber+=1 # add one to trial number
        trialarray[rowcount + 1] = trialnumber # creates array for trial number in each row of saraharray
        rowcount+=1
    return trialarray


def adjust_spines(ax,spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward',0)) # outward by 10 points
        #spine.set_smart_bounds(True)
        else:
            spine.set_color('none') # don't draw spine
    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])
    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

#functions for legends - each for a diff location
def makelegend(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles,labels, loc="baseline right", bbox_to_anchor=(1.02, 0.9), fontsize = "xx-large")
    for l in leg.get_lines():l.set_linewidth(2)
    frame  = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)

def makelegend(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles,labels, loc="baseline right", bbox_to_anchor=(0.976, 0.9), fontsize = "large")
    for l in leg.get_lines():l.set_linewidth(2)
    frame  = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)

def makelegend2(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles,labels, loc="baseline right", bbox_to_anchor=(0.976, 0.6), fontsize = "large")
    for l in leg.get_lines():l.set_linewidth(2)
    frame  = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)

def makelegend3(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles,labels, loc="baseline right", bbox_to_anchor=(0.716, 0.9), fontsize = "large")
    for l in leg.get_lines():l.set_linewidth(2)
    frame  = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)

def makelegend4(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles,labels, loc="baseline right", bbox_to_anchor=(0.716, 0.6), fontsize = "large")
    for l in leg.get_lines():l.set_linewidth(2)
    frame  = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)

def makelegend_upright(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="baseline right", bbox_to_anchor=(0.976, 0.9), fontsize="large")
    for l in leg.get_lines(): l.set_linewidth(2)
    frame = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)

def makelegend_midright(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="baseline right", bbox_to_anchor=(0.976, 0.6), fontsize="large")
    for l in leg.get_lines(): l.set_linewidth(2)
    frame = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)

def makelegend_lowright(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="baseline right", bbox_to_anchor=(0.976, 0.3), fontsize="large")
    for l in leg.get_lines(): l.set_linewidth(2)
    frame = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)

def main():
    print('-------------------------------------------------------------')
    analyse(filename='Z:\ActiveProjects\Harry\OculusVR\Data_Input\Behaviour_DataFiles/Task13_0300.h5', first_stop=True, cummulative=True)
    #analyse(filename='Z:\ActiveProjects\Harry\OculusVR\Data_Input\Behaviour_DataFiles/Task13_0300.h5', first_stop=True, cummulative=False)
    #analyse(filename='Z:\ActiveProjects\Harry\OculusVR\Data_Input\Behaviour_DataFiles/Task13_0300.h5', first_stop=False, cummulative=True)
    #analyse(filename='Z:\ActiveProjects\Harry\OculusVR\Data_Input\Behaviour_DataFiles/Task13_0300.h5', first_stop=False, cummulative=False)

    analyse(filename='Z:\ActiveProjects\Harry\OculusVR\Data_Input\Behaviour_DataFiles/Task12_0600.h5', first_stop=True, cummulative=True)
    #analyse(filename='Z:\ActiveProjects\Harry\OculusVR\Data_Input\Behaviour_DataFiles/Task12_0600.h5', first_stop=True, cummulative=False)
    #analyse(filename='Z:\ActiveProjects\Harry\OculusVR\Data_Input\Behaviour_DataFiles/Task12_0600.h5', first_stop=False, cummulative=True)
    #analyse(filename='Z:\ActiveProjects\Harry\OculusVR\Data_Input\Behaviour_DataFiles/Task12_0600.h5', first_stop=False, cummulative=False)

    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()