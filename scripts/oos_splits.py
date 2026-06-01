from __future__ import annotations

OOS_SPLITS = ["substrate_oos", "catalyst_oos", "substrate_catalyst_oos"]
OOS_IMINE = ['O=C(/N=C/c1ccc(Cl)cc1Cl)c1ccccc1']
OOS_THIOL = ['Cc1ccccc1S']
OOS_CAT = ['O=P1(O)Oc2c(-c3c(C4CCCCC4)cc(C4CCCCC4)cc3C3CCCCC3)cc3ccccc3c2-c2c(c(-c3c(C4CCCCC4)cc(C4CCCCC4)cc3C3CCCCC3)cc3ccccc23)O1', 'CC(C)c1cc(C(C)C)c(-c2cc3ccccc3c3c2OP(=O)(O)Oc2c(-c4c(C(C)C)cc(C(C)C)cc4C(C)C)cc4ccccc4c2-3)c(C(C)C)c1', 'COc1cccc(OC)c1-c1cc2ccccc2c2c1OP(=O)(O)Oc1c(-c3c(OC)cccc3OC)cc3ccccc3c1-2', 'Cc1cc(C)c(-c2cc3ccccc3c3c2OP(=O)(O)Oc2c(-c4c(C)cc(C)cc4C)cc4ccccc4c2-3)c(C)c1', 'O=P1(O)Oc2c(-c3c4ccccc4cc4ccccc34)cc3ccccc3c2-c2c(c(-c3c4ccccc4cc4ccccc34)cc3ccccc23)O1', 'O=P1(O)Oc2c(-c3ccc4ccc5cccc6ccc3c4c56)cc3ccccc3c2-c2c(c(-c3ccc4ccc5cccc6ccc3c4c56)cc3ccccc23)O1', 'O=P1(O)Oc2c(-c3ccccc3OC(F)(F)F)cc3ccccc3c2-c2c(c(-c3ccccc3OC(F)(F)F)cc3ccccc23)O1', 'CC(C)(C)c1cc(-c2cc3ccccc3c3c2OP(=O)(O)Oc2c(-c4cc(C(C)(C)C)cc(C(C)(C)C)c4)cc4ccccc4c2-3)cc(C(C)(C)C)c1', 'CC(C)(C)c1cc(-c2cc3c(c4c2OP(=O)(O)Oc2c(-c5cc(C(C)(C)C)cc(C(C)(C)C)c5)cc5c(c2-4)CCCC5)CCCC3)cc(C(C)(C)C)c1', 'Cc1ccc(-c2cc3ccccc3c3c2OP(=O)(O)Oc2c(-c4ccc(C)cc4)cc4ccccc4c2-3)cc1', 'CC(C)(C)c1ccc(-c2cc3ccccc3c3c2OP(=O)(O)Oc2c(-c4ccc(C(C)(C)C)cc4)cc4ccccc4c2-3)cc1', 'O=P1(O)Oc2c(-c3ccc(-c4ccc5ccccc5c4)cc3)cc3ccccc3c2-c2c(c(-c3ccc(-c4ccc5ccccc5c4)cc3)cc3ccccc23)O1', 'COc1ccc(-c2cc3ccccc3c3c2OP(=O)(O)Oc2c(-c4ccc(OC)cc4)cc4ccccc4c2-3)cc1', 'COCc1cccc(-c2cc3c(c4c2OP(=O)(O)Oc2c(-c5cccc(COC)c5)cc5c(c2-4)CCCC5)CCCC3)c1', 'O=P1(O)Oc2c(-c3ccccc3)cc3ccccc3c2-c2c(c(-c3ccccc3)cc3ccccc23)O1', 'C[Si](c1ccccc1)(c1ccccc1)c1cc2ccccc2c2c1OP(=O)(O)Oc1c([Si](C)(c3ccccc3)c3ccccc3)cc3ccccc3c1-2', 'O=P1(O)Oc2c(Br)cc3c(c2-c2c4c(cc(Br)c2O1)CCCC4)CCCC3', 'O=P1(O)Oc2c([Si](c3ccccc3)(c3ccccc3)c3ccccc3)cc3ccccc3c2-c2c(c([Si](c3ccccc3)(c3ccccc3)c3ccccc3)cc3ccccc23)O1', 'O=P1(O)Oc2c(Cc3ccc(C(F)(F)F)cc3C(F)(F)F)cc3ccccc3c2-c2c(c(Cc3ccc(C(F)(F)F)cc3C(F)(F)F)cc3ccccc23)O1']


def thiol_split(df):
    split = {"train": [], "substrate_oos": [], "catalyst_oos": [], "substrate_catalyst_oos": []}
    for i, row in df.iterrows():
        sub = row["Imine"] in OOS_IMINE or row["Thiol"] in OOS_THIOL
        cat = row["Catalyst"] in OOS_CAT
        split["substrate_catalyst_oos" if sub and cat else "substrate_oos" if sub else "catalyst_oos" if cat else "train"].append(i)
    split_def = {"oos_imine": OOS_IMINE, "oos_thiol": OOS_THIOL, "n_oos_catalysts": len(OOS_CAT), "oos_catalysts": OOS_CAT}
    return split, split_def
