from spikeextractors import SortingExtractor


class ShadowSortingExtractor(SortingExtractor):
    #allow the user to specify arbitrary sorting results,
    # or just copy result from an existing sorter

    def __init__(self, unit_ids=None, unit_spike_train=None, 
        samplingFrequency=None, original_sorter:SortingExtractor=None):
        # unit_spike_train is assumed to be a list of list
        super().__init__()
        self._unit_ids = unit_ids
        self._unit_spike_train = unit_spike_train
        self._sorter = original_sorter

        
   
    def get_unit_ids(self):
        if self._sorter:
            return list(self._sorter.get_unit_ids())
        else:
            return list(self._unit_ids) #make sure it is list

    def get_unit_spike_train(self, unit_id:int, start_frame=None, end_frame=None):
        if self._sorter:
            return self._sorter.get_unit_spike_train(unit_id, start_frame, end_frame)
        else:
            idx = self._unit_ids.index(unit_id)
            return self._unit_spike_train[idx]

        

