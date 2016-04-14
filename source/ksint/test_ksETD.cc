	std::string file = "../../data/ks22h001t120x64.h5";
	std::string poType = "rpo";

	MatrixXd a0;
	double T, r, s;
	int nstp;
	std::tie(a0, T, nstp, r, s) = KSreadRPO(file, poType, 1);
	
	KSETD ks(64, 22);
	ks.etdrk4->rtol = 1e-10;
	auto tmp = ks.etd(a0, T, 0.01, 1, 2, true);
	VectorXd x = ks.etdrk4->duu;
	cout << x.tail(x.size()-1).cwiseAbs().minCoeff() << endl;
	//savetxt("a.dat", x);
	
	break;
