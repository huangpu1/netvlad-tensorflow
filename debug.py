import eva_utils
import eva_init

qList, dbList = eva_init.get_List("tokyoTM/tokyoTM_train.mat")
h5File = 'index/evadata.hdf5'
eva_utils.debug(h5File, qList, dbList)
