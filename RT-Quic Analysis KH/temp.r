library(stringr)
library(readxl)
library(tidyverse)
library(ggplot2)
library(QuICAnalysis) #QuICAnalysis 1.0

# list all sub-folders in the current directory
folders <- list.dirs(path = "./RT-Quic Analysis KH/data", recursive = FALSE)

mylist <- vector(mode = "list", length = length(folders))

listnames = sub("\\./", "", folders)
names(mylist) = c(listnames)


for (i in 1:length(folders)) {
  folder <- folders[i]
  
  # List all .xlsx files in the folder
  files <- list.files(path = folder, pattern = "*.xlsx", full.names = TRUE)
  
  print(list.files(path))
  # Identify specific files based on keywords
  plate_path <- files[grepl("plate", files)]
  raw_path <- files[grepl("raw", files)]
  replicate_path <- files[grepl("replicate", files)]
  
  

  # Read the Excel files
  plate_data <- read_xlsx(plate_path)
  raw_data <- read_xlsx(raw_path)
  replicate_data <- read_xlsx(replicate_path)
  
  # Assign these dataframes to the respective sublists within "mylist"
  mylist[[i]][["plate"]] <- plate_data
  mylist[[i]][["raw"]] <- raw_data
  mylist[[i]][["replicate"]] <- replicate_data
}

mynormanalysis = vector(mode = 'list')
myanalysis = vector(mode = 'list')


for (j in 1:length(mylist)) {
  plate = mylist[[j]]$plate
  raw = mylist[[j]]$raw
  replicate = mylist[[j]]$replicate
  
  AlternativeTime = GetTime(raw)
  
  meta = GetCleanMeta(raw, plate, replicate)
  clean_raw = GetCleanRaw(meta, raw)
  
  analysis = GetAnalysis(clean_raw, sd.fold = 10, cycle_background = 10,  binw=4)
  meta.w.analysis = cbind(meta,analysis)

  print(meta.w.analysis)
  
  analysis_norm = NormAnalysis(metadata = meta, data = meta.w.analysis,
                               control_name = 'pos')
  mynormanalysis[[j]] = analysis_norm 
  myanalysis[[j]]  = meta.w.analysis
}

mypos_raf = vector(mode = 'list')
for (k in 1:length(myanalysis)) {
  sel = grep("pos", mynormanalysis[[k]]$content)
  mypos_raf[[k]] = mynormanalysis[[k]]$RAF[sel]
}

mypos_ms = vector(mode = 'list')
for (k in 1:length(myanalysis)) {
  sel = grep("pos", mynormanalysis[[k]]$content)
  mypos_ms[[k]] = mynormanalysis[[k]]$MS[sel]
}

mypos_mpr = vector(mode = 'list')
for (k in 1:length(myanalysis)) {
  sel = grep("pos", mynormanalysis[[k]]$content)
  mypos_mpr[[k]] = mynormanalysis[[k]]$MPR[sel]
}

myneg_ms = vector(mode = 'list')
for (k in 1:length(myanalysis)) {
  sel = grep("neg", mynormanalysis[[k]]$content)
  myneg_ms[[k]] = mynormanalysis[[k]]$MS[sel]
}


myneg_raf = vector(mode = 'list')
for (k in 1:length(myanalysis)) {
  sel = grep("neg", mynormanalysis[[k]]$content)
  myneg_raf[[k]] = mynormanalysis[[k]]$RAF[sel]
}


myneg_mpr = vector(mode = 'list')
for (k in 1:length(myanalysis)) {
  sel = grep("neg", mynormanalysis[[k]]$content)
  myneg_mpr[[k]] = mynormanalysis[[k]]$MPR[sel]
}

#problem is that negative controls will not follow normal distribution, even mpr...?
#maybe the sample size is too small
par(mfrow = c(2, 3))
hist(unlist(myneg_ms))
hist(unlist(myneg_raf))
hist(unlist(myneg_mpr))

hist(unlist(mypos_ms))
hist(unlist(mypos_raf))
hist(unlist(mypos_mpr))

hist(log1p(unlist(myneg_ms)))
hist(log1p(unlist(myneg_raf)))
hist(log1p(unlist(myneg_mpr)))

hist(log1p(unlist(mypos_ms)))
hist(log1p(unlist(mypos_raf)))
hist(log1p(unlist(mypos_mpr)))
