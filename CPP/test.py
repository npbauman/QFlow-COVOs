if not restart:
    mf = scf.RHF(mol)
    mf.verbose = 7
    mf.scf()
    
    mycc = cc.RCCSD(mf)
    mycc.verbose = 7
    mycc.ccsd()

eip,cip = mycc.ipccsd(nroots=1)
eea,cea = mycc.eaccsd(nroots=1)
eee,cee = mycc.eeccsd(nroots=1)

# S->S excitation
eS = mycc.eomee_ccsd_singlet(nroots=1)[0]
# S->T excitation
eT = mycc.eomee_ccsd_triplet(nroots=1)[0]