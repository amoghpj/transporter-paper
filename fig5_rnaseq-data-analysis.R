options(show.error.locations = TRUE)
myfiles <- file.path("../processed-salmon", list.files("../processed-salmon"),"quant.sf")
hanfiles <- file.path("../processed-han-salmon", list.files("../processed-han-salmon"),"quant.sf")
# files <- c("../processed-salmon-flat/TRA00209271.sf","../processed-salmon-flat/TRA00209272.sf")

library(GenomicFeatures)
library(tximport)
library(DESeq2)
library(ggplot2)

txdb <- makeTxDbFromGFF("../reference/GCF_000146045.2_R64_genomic.gff")
hxts <- c("HXT1","HXT2", "HXT3","HXT4","HXT5","HXT6","HXT7","HXT8", "HXT9","HXT10","HXT11","HXT13","HXT14","HXT15","HXT16", "HXT17")
k <- keys(txdb)

tx2gene <- select(txdb, keys=k, columns="TXNAME", keytype="GENEID")
tx2gene <- tx2gene[, c("TXNAME", "GENEID")]

mytxi <- tximport(myfiles, type="salmon", tx2gene=tx2gene)


idcols<-c("2-glc-cd103-r1","02-glc-cd103-r1","002-glc-cd103-r1","2-fru-cd103-r1","2-gal-cd103-r1","2-raf-cd103-r1","2-mal-cd103-r1","2-ace-cd103-r1","2-glc-sh07-r1","02-glc-sh07-r1","002-glc-sh07-r1","2-fru-sh07-r1","2-gal-sh07-r1","2-raf-sh07-r1","2-mal-sh07-r1","2-ace-sh07-r1","2-glc-cd103-r2","02-glc-cd103-r2","002-glc-cd103-r2","2-fru-cd103-r2","2-gal-cd103-r2","2-raf-cd103-r2","2-mal-cd103-r2","2-ace-cd103-r2","2-glc-sh07-r2","02-glc-sh07-r2","002-glc-sh07-r2","2-fru-sh07-r2","2-gal-sh07-r2","2-raf-sh07-r2","2-mal-sh07-r2","2-ace-sh07-r2","2-glc-cd103-r3","02-glc-cd103-r3","002-glc-cd103-r3","2-fru-cd103-r3","2-gal-cd103-r3","2-raf-cd103-r3","2-mal-cd103-r3","2-ace-cd103-r3","2-glc-sh07-r3","02-glc-sh07-r3","002-glc-sh07-r3","2-fru-sh07-r3","2-gal-sh07-r3","2-raf-sh07-r3","2-mal-sh07-r3","2-ace-sh07-r3")

uniqueconds<-c("2-glc-cd103","02-glc-cd103","002-glc-cd103","2-fru-cd103","2-gal-cd103","2-raf-cd103","2-mal-cd103","2-ace-cd103","2-glc-sh07","02-glc-sh07","002-glc-sh07","2-fru-sh07","2-gal-sh07","2-raf-sh07","2-mal-sh07","2-ace-sh07","2-glc-cd103","02-glc-cd103","002-glc-cd103","2-fru-cd103","2-gal-cd103","2-raf-cd103","2-mal-cd103","2-ace-cd103","2-glc-sh07","02-glc-sh07","002-glc-sh07","2-fru-sh07","2-gal-sh07","2-raf-sh07","2-mal-sh07","2-ace-sh07","2-glc-cd103","02-glc-cd103","002-glc-cd103","2-fru-cd103","2-gal-cd103","2-raf-cd103","2-mal-cd103","2-ace-cd103","2-glc-sh07","02-glc-sh07","002-glc-sh07","2-fru-sh07","2-gal-sh07","2-raf-sh07","2-mal-sh07","2-ace-sh07")

repcols<-c("r1","r1","r1","r1","r1","r1","r1","r1","r1","r1","r1","r1","r1","r1","r1","r1","r2","r2","r2","r2","r2","r2","r2","r2","r2","r2","r2","r2","r2","r2","r2","r2","r3","r3","r3","r3","r3","r3","r3","r3","r3","r3","r3","r3","r3","r3","r3","r3")

nutcols<-c("2-glc","02-glc","002-glc","2-fru","2-gal","2-raf","2-mal","2-ace","2-glc","02-glc","002-glc","2-fru","2-gal","2-raf","2-mal","2-ace","2-glc","02-glc","002-glc","2-fru","2-gal","2-raf","2-mal","2-ace","2-glc","02-glc","002-glc","2-fru","2-gal","2-raf","2-mal","2-ace","2-glc","02-glc","002-glc","2-fru","2-gal","2-raf","2-mal","2-ace","2-glc","02-glc","002-glc","2-fru","2-gal","2-raf","2-mal","2-ace")

straincols<-c("cd103","cd103","cd103","cd103","cd103","cd103","cd103","cd103","sh07","sh07","sh07","sh07","sh07","sh07","sh07","sh07","cd103","cd103","cd103","cd103","cd103","cd103","cd103","cd103","sh07","sh07","sh07","sh07","sh07","sh07","sh07","sh07","cd103","cd103","cd103","cd103","cd103","cd103","cd103","cd103","sh07","sh07","sh07","sh07","sh07","sh07","sh07","sh07")

sampleTable <- data.frame(ids = idcols,
                          #reps=factor(repcols),
                          strains=factor(straincols),
                          uconds=factor(uniqueconds),
                          nutrient=factor(nutcols))
# print(sampleTable)
print("Starting Han's datasets...")
rownames(sampleTable) <- colnames(mytxi$counts)
mydds<- DESeqDataSetFromTximport(mytxi, sampleTable, design=~ strains + nutrient)

idcols<-c("2-glc-a28y-r1","02-glc-a28y-r1","002-glc-a28y-r1","2-raf-a28y-r1","2-ace-a28y-r1","2-fru-a28y-r1","2-mal-a28y-r1","2-gal-a28y-r1","2-glc-a28y-r2","02-glc-a28y-r2","002-glc-a28y-r2","2-raf-a28y-r2","2-ace-a28y-r2","2-fru-a28y-r2","2-mal-a28y-r2","2-gal-a28y-r2")

uniqueconds<-c("2-glc-a28y","02-glc-a28y","002-glc-a28y","2-raf-a28y","2-ace-a28y","2-fru-a28y","2-mal-a28y","2-gal-a28y","2-glc-a28y","02-glc-a28y","002-glc-a28y","2-raf-a28y","2-ace-a28y","2-fru-a28y","2-mal-a28y","2-gal-a28y")

nutcols<-c("2-glc","02-glc","002-glc","2-raf","2-ace","2-fru","2-mal","2-gal","2-glc","02-glc","002-glc","2-raf","2-ace","2-fru","2-mal","2-gal")
straincols<-c("a28y","a28y","a28y","a28y","a28y","a28y","a28y","a28y","a28y","a28y","a28y","a28y","a28y","a28y","a28y","a28y")
hantxi <- tximport(hanfiles, type="salmon", tx2gene=tx2gene)
print("Might be failing before this")
rownames(sampleTable) <- colnames(hantxi$counts)
sampleTable <- data.frame(ids = idcols,
                          strains=straincols,
                          uconds=factor(uniqueconds),
                          nutrient=factor(nutcols))
sampleTable
handds<- DESeqDataSetFromTximport(hantxi, sampleTable, design=~nutrient)
print("Han DDS created")
dds <- cbind(mydds, handds)
dds
hxtexp <- dds[hxts]
hxtexpcomb <- collapseReplicates(hxtexp, hxtexp$uconds)
hxtexpcomb
heclog <- vst(hxtexpcomb, nsub=nrow(hxtexpcomb))  # VarianceStabilizingTransformation
## what does intgroup mean?
p <- plotPCA(heclog, intgroup=c("uconds","strains","nutrient"))
write.csv(p$data, "expression-PCA.csv")
ggplot(p$data, aes(PC1,PC2,label=uconds,color=nutrient)) +
    #geom_point(aes(color=nutrient))# + 
    geom_text(aes(color=nutrient))

ggsave("pca-hxt-1.png",
       plot=last_plot(),
       dpi=200)
