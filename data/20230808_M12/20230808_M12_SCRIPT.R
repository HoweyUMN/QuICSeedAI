library(readxl)
library(tidyverse)
library(ggplot2)
library(QuICAnalysis)

#read files 
plate = read_xlsx('20230808_M12_plate.xlsx')
raw = read_xlsx('20230808_M12_raw.xlsx')
replicate = read_xlsx('20230808_M12_replicate.xlsx')

#get run time
AlternativeTime = GetTime(raw)

#get metadata
meta = GetCleanMeta(raw, plate, replicate)
Instrument_number <- str_extract('20230808_M12_plate', "[A-Z]\\d+") #str_extract(file_names, "[T]\\d+") 
meta$instrument = Instrument_number

#clean fluorescence data from MARS, there are ways to extract fluorescence readings only from samples of interest
clean_raw = GetCleanRaw(meta, raw)

#plot negative controls 
PlotRawSingle(clean_raw, 'neg')

#plot multiple samples, here positive and negative controls 
samples = c('pos', 'neg')
PlotRawMulti(clean_raw, samples) 

#if you were to do 65c cut-off on a 97c data
 cycle_total = 65
 clean_raw.65 = clean_raw[1:cycle_total, ]
 PlotRawMulti(clean_raw.65, samples)

#get analytic metrics of the run 
analysis = GetAnalysis(clean_raw, sd.fold = 10, cycle_background = 4,  binw=4)

#combine metadata and analysis
meta.w.analysis = cbind(meta,analysis)

#normalize analysis with positive controls numbers, given that the same positive controls were used 
analysis_norm = NormAnalysis(metadata = meta, data = meta.w.analysis,
                             control_name = 'pos')

#spread out data, can be used for ploting in PRISM
norm_analysis_spread = GetSpreadData(analysis_norm)

#can apply different statistical tests with different parameters
stats_norm = GetStats(norm_analysis_spread, 'neg', test = 'wilcox', tail = 'greater') 
stats_norm_new = GetStats(norm_analysis_spread, 'neg', test = 'yuen', tail = 'greater')

# analysis_spread = GetSpreadData(meta.w.analysis)
# stats = GetStats(analysis_spread, 'neg')

# get a clean results, default is when all three metrics are positive 
result = GetCombinedResult(stats_norm, meta)
sel = filter(result, result == '*')
samples = rownames(sel)
PlotRawMulti(clean_raw, samples) #pos, 71, 77, 78

result_new = GetCombinedResult(stats_norm_new, meta)
sel = filter(result_new, result == '*')
samples = rownames(sel)
PlotRawMulti(clean_raw, samples)
PlotRawSingle(clean_raw, '64')
PlotRawMulti(clean_raw, c('pos','64','96','91'))
PlotRawSingle(clean_raw, '91')

#change what "positive" is 
sel = filter(result, metric_count == 2)
samples = rownames(sel)
samples = c( samples)
#png("PositiveSamplePlot.png", height = 4, width = 7, units = "in", res = 300)
PlotRawMulti(clean_raw, c('pos',samples))
#dev.off()

sel = filter(result_new, metric_count == 2)
samples = rownames(sel)
samples = c( samples)
PlotRawMulti(clean_raw, samples)
#write.csv(result, 'result.csv')


#png("MPR.png", height = 4, width = 7, units = "in", res = 300)
ggplot(meta.w.analysis, aes(x=content, y=MPR)) +  #can change to many different things
  geom_boxplot() +
  geom_jitter(shape=16, position=position_jitter(0.2)) +
  theme_bw()+
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) 
#dev.off()



