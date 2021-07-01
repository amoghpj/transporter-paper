# AUTHOR: Amogh Jalihal
# This workflow file is used to process the RNAseq data used in Figure 5

PATH_TRANSCRIPTOME = "../reference/GCF_000146045.2_R64_rna.fna"
PATH_GENOME = "../reference/GCF_000146045.2_R64_genomic.fna"
PATH_ANNOTATION = "../reference/GCF_000146045.2_R64_genomic.gff"
PATH_ANNOTATION2 = "../reference/Saccharomyces_cerevisiae.R64-1-1.104.gtf"
PATH_TRANSCRIPTOME_INDEX = "../index/yeast_index"
PATH_TRANSCRIPTOME_INDEX_SALMON = "../yeast_index_salmon"
PATH_PREFIX = "../FC_06725/Unaligned_1234_PF_mm1/"
PATH_RAWDATA = PATH_PREFIX + "Data/Project_jamogh/"
SAMPLE_IDS = PATH_PREFIX + "FC_06725_1234_PF_mm1_jamogh_manifest.csv"

### Han's data
HAN_PATH_PREFIX = "../../20191122_transporter/Unaligned_1234_PF_mm1/"
HAN_PATH_RAWDATA = HAN_PATH_PREFIX + "Data/Project_SAB15Seq/"
HAN_SAMPLE_ID_PATH = HAN_PATH_PREFIX + "samples.txt"
import pandas as pd

def get_han_sample_ids():
    df = pd.read_csv(SAMPLE_IDS,header=0,sep=" ")
    "LIB044265_" + "TRA00169311_S16_L004_R1.fastq.bz2"
    libpool = "LIB044265"
    # Produced by hand
    libnames = ["TRA00169296",
                "TRA00169297",
                "TRA00169298",
                "TRA00169299",
                "TRA00169300",
                "TRA00169301",
                "TRA00169302",
                "TRA00169303",
                "TRA00169304",
                "TRA00169305",
                "TRA00169306",
                "TRA00169307",
                "TRA00169308",
                "TRA00169309",
                "TRA00169310",
                "TRA00169311"]
    return((libpool, libnames))
    
def get_sample_ids():
    df = pd.read_csv(SAMPLE_IDS,header=0)
    libpool = df[" Library Pool ID"][0]
    libnames = df[" Library Id"]
    return((libpool, libnames.values))

LIBPOOL, LIBNAMES = get_sample_ids()
LIBNAMES = sorted(list(set(LIBNAMES)))

HAN_LIBPOOL, HAN_LIBNAMES = get_han_sample_ids()
HAN_LIBNAMES = sorted(list(set(HAN_LIBNAMES)))

rule all:
    input:
        "allcounts_hs_fc.csv",
        "allcounts_salmon.csv",
        "allcounts_han_salmon.csv"
        #[f"../processed-salmon/{sample}/quant.sf" for sample in LIBNAMES]
    # run:
    #     for outf in input:
    #         shell(f"cp {{outf}} ../../../projects/.data/")
         
rule build_reference_salmon:
    input:
        PATH_TRANSCRIPTOME
    output:
        directory(PATH_TRANSCRIPTOME_INDEX_SALMON)
    shell:
        "salmon index -t {input} -i {output}"
        
rule build_salmon_map:
    input:
        [PATH_RAWDATA + LIBPOOL + f"_{sample}_S{i+1}_L00{j}_R1.fastq.bz2" for j in range(1,5) for i, sample in enumerate(LIBNAMES)],
        PATH_TRANSCRIPTOME_INDEX_SALMON
    output:
         [f"../processed-salmon/{sample}/quant.sf" for sample in LIBNAMES]        
    run:
        for i,sample in enumerate(LIBNAMES):
            infilestr = " ".join(["<(bunzip2 -c \"" +  PATH_RAWDATA + LIBPOOL + f"_{sample}_S{i+1}_L00{j}_R1.fastq.bz2\")" for j in range(1,5)])
            shell(f"salmon quant -i \"{PATH_TRANSCRIPTOME_INDEX_SALMON}\" -l A  -r {{infilestr}} -p 8   --validateMappings -o ../processed-salmon/{{sample}}") #-g {PATH_ANNOTATION}

rule build_salmon_map_han:
    input:
        [HAN_PATH_RAWDATA + HAN_LIBPOOL + f"_{sample}_S{i+1}_L00{j}_R1.fastq.bz2" for j in range(1,5) for i, sample in enumerate(HAN_LIBNAMES)],
        PATH_TRANSCRIPTOME_INDEX_SALMON
    output:
         [f"../processed-han-salmon/{sample}/quant.sf" for sample in HAN_LIBNAMES]        
    run:
        for i,sample in enumerate(HAN_LIBNAMES):
            infilestr = " ".join(["<(bunzip2 -c \"" +\
                                  HAN_PATH_RAWDATA + HAN_LIBPOOL +\
                                  f"_{sample}_S{i+1}_L00{j}_R1.fastq.bz2\")" for j in range(1,5)])
            shell(f"salmon quant -i \"{PATH_TRANSCRIPTOME_INDEX_SALMON}\" -l A  -r {{infilestr}} -p 8   --validateMappings -o ../processed-han-salmon/{{sample}}") #-g {PATH_ANNOTATION}            
            
rule get_counts_salmon:
    input:
         [f"../processed-salmon/{sample}/quant.sf" for sample in LIBNAMES]
    output:
        "allcounts_salmon.csv"
    run:
        import pandas as pd
        dflist = []
        df1 = pd.read_csv(input[0],sep='\t',index_col=0)
        reads = df1.NumReads
        columns=['Length', 'EffectiveLength']
        columns.extend(LIBNAMES)
        df = pd.DataFrame(columns=columns, index=df1.index)
        df.Length = df1.Length
        df.EffectiveLength = df1.EffectiveLength
        df[LIBNAMES[0]] = df1.TPM
        for inf,sample in zip(input[1:], LIBNAMES[1:]):
            df[sample] = pd.read_csv(inf,sep='\t',index_col=0).TPM
        print(output[0])
        df.to_csv(output[0])


rule get_counts_salmon_han:
    input:
         [f"../processed-han-salmon/{sample}/quant.sf" for sample in HAN_LIBNAMES]
    output:
        "allcounts_han_salmon.csv"
    run:
        import pandas as pd
        dflist = []
        df1 = pd.read_csv(input[0],sep='\t',index_col=0)
        reads = df1.NumReads
        columns=['Length', 'EffectiveLength']
        columns.extend(HAN_LIBNAMES)
        df = pd.DataFrame(columns=columns, index=df1.index)
        df.Length = df1.Length
        df.EffectiveLength = df1.EffectiveLength
        df[HAN_LIBNAMES[0]] = df1.TPM
        for inf,sample in zip(input[1:], HAN_LIBNAMES[1:]):
            df[sample] = pd.read_csv(inf,sep='\t',index_col=0).TPM
        print(output[0])
        df.to_csv(output[0])        
            
rule build_reference_hisat:
    input:
        PATH_GENOME
    output:
        index=expand("{ind}.{num}.ht2",num=list(range(1,9)),ind=PATH_TRANSCRIPTOME_INDEX)
    shell:
        "hisat2-build {input} {PATH_TRANSCRIPTOME_INDEX}"

rule build_samfiles:
    input:
        [PATH_RAWDATA + LIBPOOL + f"_{sample}_S{i+1}_L00{j}_R1.fastq.bz2" for j in range(1,5) for i, sample in enumerate(LIBNAMES)],
        [f"{PATH_TRANSCRIPTOME_INDEX}.{i}.ht2" for i in range(1,9)]
    output:
        expand("../processed/{sample}.bam",sample=LIBNAMES)
    run:
        for i,sample in enumerate(LIBNAMES):
            infilestr = ",".join([PATH_RAWDATA + LIBPOOL + f"_{sample}_S{i+1}_L00{j}_R1.fastq.bz2" for j in range(1,5)])
            shell(f"hisat2 -p 10 -t -x {PATH_TRANSCRIPTOME_INDEX} -U {{infilestr}} | samtools view --threads 8 -bS -o ../processed/{{sample}}.bam")

rule get_counts_fc:
    input:
        expand("../processed/{samples}.bam", samples=LIBNAMES)
    output:
        "allcounts_hs_fc.csv"
    run:
        shell("featureCounts -T 4 -t gene -g locus_tag -a {PATH_ANNOTATION} -o {output} {input}")
        
rule clean:
    shell:
        "rm -rf  allcounts* ../processed/ ../index/*ht2 ./*.ht2 ../processed-salmon/ ../yeast_index_salmon/"
