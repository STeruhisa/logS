import argparse
import itertools
import numpy as np
import pandas as pd
from rdkit import Chem

def parse_args():
    parser = argparse.ArgumentParser(description='Get IG_features')

    parser.add_argument('--IG_in',  default='', type=str, help='input IG file')
    parser.add_argument('--sdf',    default='', type=str, help='sdf file')
    parser.add_argument('--IG_out', default='', type=str, help='output IG file')

    return parser.parse_args()

def hansch_substruct():

    slist = []
#
#   -SO2CF3 [1]
#
    slist.append(Chem.MolFromSmarts('[SX4](=[OX1])(=[OX1])[CX4]([F])([F])([F])'))
#
#   -SO2CH3 [2]
#
    slist.append(Chem.MolFromSmarts('[SX4](=[OX1])(=[OX1])[CH3]([H])([H])([H])'))
#
#   -NHCOCH3 [3]
#
    slist.append(Chem.MolFromSmarts('[NX3H]([H])[CX3](=[OX1])[CH3]([H])([H])([H])'))
#
#   -NHCONH2 [4]
#
    slist.append(Chem.MolFromSmarts('[NX3H]([H])[CX3](=[OX1])[NH2]([H])([H])'))
#
#   -SCH3 [5]
#
    slist.append(Chem.MolFromSmarts('[SD2][CH3]([H])([H])([H])'))
#
#   -COCH3 [6]
#
    slist.append(Chem.MolFromSmarts('[CX3](=[OX1])[CH3]([H])([H])([H])'))
#
#   -OCF3 [7]
#
    slist.append(Chem.MolFromSmarts('[OX2][CX4]([F])([F])([F])'))
#
#   -OCH3 [8]
#
    slist.append(Chem.MolFromSmarts('[OX2][CH3]([H])([H])([H])'))
#
#   -tBu [9]
#
    slist.append(Chem.MolFromSmarts('[CX4]([CH3]([H])([H])([H]))([CH3]([H])([H])([H]))([CH3]([H])([H])([H]))'))
#
#   -sBu [10]
#
    slist.append(Chem.MolFromSmarts('[CX4H1]([H])([CH3]([H])([H])([H]))([CX4H2]([H])([H])([CH3]([H])([H])([H])))'))
#
#   -nBu [11]
#
    slist.append(Chem.MolFromSmarts('[CX4H2]([H])([H])[CX4H2]([H])([H])[CX4H2]([H])([H])[CX4H3]([H])([H])([H])'))
#
#   -iPr [12]
#
    slist.append(Chem.MolFromSmarts('[CX4H1]([H])([CH3]([H])([H])([H]))([CH3]([H])([H])([H]))'))
#
#   -nPr [13]
#
    slist.append(Chem.MolFromSmarts('[CX4H2]([H])([H])[CX4H2]([H])([H])[CX4H3]([H])([H])([H])'))
#
#   -Et [14]
#
    slist.append(Chem.MolFromSmarts('[CX4H2]([H])([H])[CX4H3]([H])([H])([H])'))
#
#   -COOH [15]
#
    slist.append(Chem.MolFromSmarts('[CX3](=O)[OX2H1]([H])'))
#
#   -OH [16]
#
    slist.append(Chem.MolFromSmarts('[OX2H]([H])'))
#
#   -CN [17]
#
    slist.append(Chem.MolFromSmarts('[CX2](#[NX1])'))
#
#   -CF3 [18]
#
    slist.append(Chem.MolFromSmarts('[CX4]([F])([F])([F])'))
#
#   -CH3 [19]
#
    slist.append(Chem.MolFromSmarts('[CX4H3]([H])([H])([H])'))
#
#   -I [20]
#
    slist.append(Chem.MolFromSmarts('[I]'))
#
#   -Br [21]
#
    slist.append(Chem.MolFromSmarts('[Br]'))
#
#   -Cl [22]
#
    slist.append(Chem.MolFromSmarts('[Cl]'))
#
#   -F [23]
#
    slist.append(Chem.MolFromSmarts('[F]'))


    return slist

def prp_molnum_dict(sdf_path):

    sdf_sup = Chem.ForwardSDMolSupplier(sdf_path,removeHs=False,sanitize=False)

    sub_struct_list = hansch_substruct()
    num_list        = len(sub_struct_list)

    moldict = {}
    numdict = {}

    for x in sdf_sup:
        x.UpdatePropertyCache(strict = False)

        mol_name = x.GetProp('_Name').strip('_')
        nat      = x.GetNumAtoms()

        atom_indx = np.zeros(nat,dtype=int)
        num_indx  = np.zeros(num_list,dtype=int)

        for i, sub_struct in enumerate(sub_struct_list):

            if x.HasSubstructMatch(sub_struct):
                matches = list(x.GetSubstructMatches(sub_struct))

                for match in matches:
#
#                   first : check whether the atom is assigned
#
                    icheck = 0
                    for atom in list(match):
                        if atom_indx[atom] != 0:
                            icheck = 1
#
#                   second : store the data
#
                    if icheck == 0:
                        for atom in list(match):
                            atom_indx[atom] = i + 1
               
                        num_indx[i] = num_indx[i] + 1

        moldict[mol_name] = atom_indx
        numdict[mol_name] = num_indx             

    return moldict, numdict, num_list

def main(args):
#
#   input & output files
#
    IG_in = args.IG_in
    IG_out = args.IG_out
#
#   sdf file
#
    moldict, numdict, num_substract = prp_molnum_dict(args.sdf)
#
#   get nat_list, CID_list, and istart_list
#
    istart_list = []
    CID_list = []
    count = -1
    nmol = -1
    with open(IG_in) as f:
        for line in f:
            count += 1
            tmp = line.split()
            if tmp[0] == '#':
                nmol += 1
                istart_list.append(count)
                CID_list.append(tmp[1])

            if count == 1:
               tmp_str = line.rstrip("\n").split()
               nfeatures = len(tmp_str)

    nat_list = []
    for imol in range(nmol):
       nat = istart_list[imol+1] - istart_list[imol] - 1
       nat_list.append(nat)

    nat = count - istart_list[nmol]
    nat_list.append(nat)
#
#   store data -> IG_features
#

    IG_substruct = [[] for i in range(num_substract)]
    CAS_substruct = [[] for i in range(num_substract)]

    with open(IG_in) as f:
        lines = f.readlines()

        for imol, istart in enumerate(istart_list):
            nat = nat_list[imol]
            CID = CID_list[imol]

            tmp_sum = np.zeros(num_substract)
            for iat, indx in enumerate(moldict[CID]):
                if indx != 0:
                    tmp_str = lines[istart+iat+1].rstrip("\n").split()
                    tmp_float = np.array([float(d) for d in tmp_str])

                    tmp_sum[indx-1] = tmp_sum[indx-1] + np.sum(tmp_float)

            for i in range(num_substract):
                for j in range(numdict[CID][i]):
                    IG_substruct[i].append(tmp_sum[i]/numdict[CID][i])

                if numdict[CID][i] != 0:
                    CAS_substruct[i].append(CID)

#
#   get average
#
    IG_avg = np.zeros(num_substract)   
    for i in range(num_substract):
        ntot = len(IG_substruct[i])
 
        if ntot != 0:
            tmp = 0.0
            for x in IG_substruct[i]:
                tmp = tmp + x

            IG_avg[i] = tmp/float(ntot)
#
#   get RMSD
#
    IG_RMSD = np.zeros(num_substract)
    for i in range(num_substract):
        ntot = len(IG_substruct[i])

        if ntot != 0:
            tmp = 0.0
            for x in IG_substruct[i]:
                tmp = tmp + (x - IG_avg[i])**2

            IG_RMSD[i] = tmp/float(ntot)

    print(CAS_substruct[15])
    print(IG_substruct[15])
    for x in IG_substruct[15]:
        print(x) 

#   data list
#
    data_list = []
    for i in range(num_substract):
        data = [i+1, IG_avg[i], IG_RMSD[i]]

        data_list.append(data)

    df_list = pd.DataFrame(data_list,columns=["indx","avg","RMSD"])

    print(df_list)

if __name__ == "__main__":
    args = parse_args()
    main(args)
