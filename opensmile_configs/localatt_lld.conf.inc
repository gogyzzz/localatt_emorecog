///////////////////////////////////////////////////////////////////////////////////////
///////// > openSMILE configuration file for data output <           //////////////////
/////////                                                            //////////////////
///////// (c) 2014-2016 audEERING,                                   //////////////////
/////////     All rights reserverd. See file COPYING for details     //////////////////
///////////////////////////////////////////////////////////////////////////////////////


/*
   This file can be included as data output file for standard feature
   extraction configuration files. It provides commandline options
   for the batch extraction GUI, and supports LLD and Functionals (summaries)
   saving.

   It requires the main extrator configuration file to provide the following
   data memory levels:  lld, lld_de, and func
 */

;;;;;;;;; output LLD features to CSV ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

[componentInstances:cComponentManager]
instance[lldhtksink].type=cHtkSink

;filename=\cm[output(O){mfcc.htk}:name of MFCC output filename (HTK format)]

[lldhtksink:cHtkSink]
reader.dmLevel = lld;lld_de
filename=\cm[lldhtkoutput(O){output.lld.sink.htk}:output HTK binary file for LLD, disabled by default ?, only written if filename given]
append = \cm[appendhtklld{0}:set to 1 to append to the LLD output htk file, default is not to append]
  ; this is broken for HTK sink...
errorOnNoOutput = 0  
parmKind = 9
