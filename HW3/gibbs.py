import numpy as np 
import sys

#this function is imported from HW2
def parse_FASTA_file(fasta_file):
    sequences = [] # the list we will store the sequences 
    temp_list = [] # temporary list for looping each line 
    
    for line in fasta_file: 
        line = line.strip()#remove the /m
        if line.startswith('>'): #initiate loop with searching for the headers 
            if temp_list: #if the list is not empty, append the elements to the sequences list
                sequences.append(''. join(temp_list))
            temp_list=[]   # empty the list after you are done, so that it can loop the next set following a header confrmation 
        else:
            temp_list.append(line) #return the line element to loop it back so it can go back to the original
    if temp_list:  
        sequences.append(''.join(temp_list)) #make sure to include the last element 
    return sequences    

    
class SimpleGibbsSampler:
    def __init__(self, seqs, motif_len, lang, lang_prob, seed):#initialize the class with the sequences and the scoring system
        self.seqs = seqs
        self.motif_len = motif_len
        self.lang = lang
        self.lang_prob = lang_prob
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        
    def pick_init_positions(self):  
        random_idxs = [] #initialize the index list for the new list of seqences
        for seq in self.seqs:
            random_idxs.append(self.random_state.randint(len(seq)-self.motif_len+1)) #randomly select the index of the motif, esnure the motif alway returns a motif of length 6
        return random_idxs
        
    def motifs_finder(self,random_idxs,seqs):
        motifs = [] #initialize the list of motifs aftr indexing
        for i in range(len(random_idxs)):
            idx = random_idxs[i] 
            motifs.append(self.seqs[i][idx:idx+self.motif_len])   #append the motif to the list of motifs
        return motifs 
    
    def build_counts_matrix(self, motifs):
        alphabet = self.lang #define the alphabet of nucleotides
        counts_matrix_zero = np.zeros((len(alphabet), self.motif_len)) #initialize the count matrix with zeros
        counts_matrix = [[val+1 for val in row] for row in counts_matrix_zero] #add the pseudocount of 1 to each element in the count matrix
        for seqs in motifs: #look through each sequence in the list of motifs
            for j in range(len(seqs)): #look through each nucleotide in the sequence
                for i in range(len(alphabet)): #see if the nucleotides match 
                    if seqs[j] == alphabet[i]:#if they match, add 1 to the count matrix
                        counts_matrix[i][j] +=1
        return np.array(counts_matrix)
    
    def build_prop_matrix(self, counts_matrix):
        divider = np.array([np.sum(counts_matrix,0)])#sum the counts matrix along the columns
        PSSM = (counts_matrix/divider)/self.lang_prob #get the frequency of each nucleotide and then dicide by the frequency of the nucleotides
        return PSSM
    
    def get_scoring_motifs(self, seq):
        seqs_for_scoring= []#initialize the list of sequences we will use for scoring
        for i in range(len(seq) - self.motif_len +1):#loop through the sequence to get the sequences for scoring
            seqs_for_scoring.append(seq[i:i+self.motif_len])#append the sequences to the list of sequences for scoring based on the length of the motif
        return seqs_for_scoring
    
    def score_seq_windows(self, seq, PSSM):
        seqs_for_scoring = []#initialize the list of sequences we will use for scoring
        for i in range(len(seq) - self.motif_len +1):#loop through the sequence to get the sequences for scoring
            seqs_for_scoring.append(seq[i:i+self.motif_len])#append the sequences to the list of sequences for scoring based on the length of the motif
        alphabet = self.lang
        seq_val = []
        for seq in seqs_for_scoring:
            counter = 1 #initialize the counter that will be used to multiply the values of the PSSM
            for j in range(len(seq)):
                for i in range(len(alphabet)):
                    if seq[j] == alphabet[i]:# match the nucleotides in the sequence to the one in the sequnece 
                        counter = counter * PSSM[i][j] #multiply for the corresponding PSSM value
            seq_val.append(counter)  
        return seq_val  
    
    def get_msa(self, index_list):
        MSA = []
        for i in range(len(self.seqs)):
            MSA.append(self.seqs[i][index_list[i]:index_list[i]+ self.motif_len])
        return MSA
    
    def run_sampler(self):
        itteration = 0 #initialize the itteration counter
        random_idxs = self.pick_init_positions() #initialize the random indexes for the motifs
        random_idxs_copy = random_idxs[:] #copy the random indexes for the motifs so that we can have a copy of each prior itteration
        while True: #initiate an infinite loop to run the sampler
            itteration += 1 #count the number of itterations
            random_index = self.random_state.randint(len(self.seqs)) #select a random index to remove a sequence from the list of sequences
            s_star = self.seqs[random_index] #select a random sequence to remove from the list of sequences
            mod_random_idxs = random_idxs[:random_index] + random_idxs[random_index+1:] #remove the index of the sequence from the list of indexes for the motifs
            mod_seqs = self.seqs[:random_index] + self.seqs[random_index+1:]#remove the sequence from the list of sequences for initiating motifs 
            motifs = self.motifs_finder(mod_random_idxs, mod_seqs) #return the motifs for scoring
            counts_matrix = self.build_counts_matrix(motifs) #return the counts matrix
            PSSM = self.build_prop_matrix(counts_matrix) #return the propesnity matrix
            scores = self.score_seq_windows(s_star, PSSM) #score all the motifs of length 6 from the s*.
            max_idx = scores.index(max(scores))#return the index of the motif with the highest score
            random_idxs[random_index] = max_idx  
            if random_idxs == random_idxs_copy: #set the convergence criteria for the while loop to stop
                break
            random_idxs_copy = random_idxs[:] #copy the indexes of this run
        MSA = self.get_msa(random_idxs) #return the common motifs we discovered
        print(f'The number of itterations is: {itteration}') 
        print(f'The last sequence removed is:{s_star}')
        print(f'The last propensity matrix is:{PSSM}')
        print(f'The common motifs are: {MSA}')
        return MSA
    

def main(): #main function to run the program
    seqs = parse_FASTA_file(sys.stdin) #add the standin for the FASTA files
    alphabet = ['A', 'C', 'G', 'T'] #define the alphabet of nucleotides
    sgs = SimpleGibbsSampler(seqs, 6, alphabet, 0.25, 14) #initialize the class with inputs 
    msa = sgs.run_sampler()#run the sampler
    msa#return the itteration, motifs, last s*
    
if __name__ == '__main__': #run the main function oin comand console
    main()


    