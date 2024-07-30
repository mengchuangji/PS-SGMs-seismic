def str_divne(num, den, Niter, rect, ndat, eps_dv, eps_cg, tol_cg,verb):
	#str_divne: N-dimensional smooth division rat=num/den		  
	#This is a subroutine from the seistr package (https://github.com/chenyk1990/seistr)
	#
	#Ported to Python by Yangkang Chen, 2022
	#
	#INPUT
	#num: numerator
	#den: denominator
	#Niter: number of iterations
	#rect: triangle radius [ndim], e.g., [5,5,1]
	#ndat: data dimensions [ndim], e.g., [n1,n2,1]
	#eps_dv: eps for divn  (default: 0.01)
	#eps_cg: eps for CG	(default: 1)
	#tol_cg: tolerence for CG (default: 0.000001)
	#verb: verbosity flag
	#
	#OUTPUT
	#rat: output ratio
	# 
	#Reference
	#H. Wang, Y. Chen, O. Saad, W. Chen, Y. Oboue, L. Yang, S. Fomel, and Y. Chen, 2021, A Matlab code package for 2D/3D local slope estimation and structural filtering: in press.
	import numpy as np
	n=num.size
	
	ifhasp0=0
	p=np.zeros(n)
	
	num=num.reshape(n,order='F')
	den=den.reshape(n,order='F')
	
	if eps_dv > 0.0:
		for i in range(0,n):
			norm = 1.0 / np.hypot(den[i], eps_dv);
			num[i] = num[i] * norm;
			den[i] = den[i] * norm;
	norm=sum(den*den);
	if norm == 0.0:
		rat=np.zeros(n);
		return rat
	norm = np.sqrt(n / norm);
	num=num*norm;
	den=den*norm;
		
	par_L={'nm':n,'nd':n,'w':den}
	par_S={'nm':n,'nd':n,'nbox':rect,'ndat':ndat,'ndim':3}
	
	
	rat = str_conjgrad('NULL', str_weight_lop, str_trianglen_lop, p, 'NULL', num, eps_cg, tol_cg, Niter,ifhasp0,[],par_L,par_S,verb);
	rat=rat.reshape(ndat[0],ndat[1],ndat[2],order='F')

	return rat


def str_weight_lop(din,par,adj,add):
	# str_weight_lop: Weighting operator (verified)
	# 
	# Ported to Python by Yangkang Chen, 2022
	# 
	# INPUT
	# din: model/data
	# par: parameter
	# adj: adj flag
	# add: add flag
	# OUTPUT
	# dout: data/model
	# 
	import numpy as np
	nm=par['nm'];
	nd=par['nd'];
	w=par['w'];

	if adj==1:
		d=din;
		if 'm' in par and add==1:
			m=par['m'];
		else:
			m=np.zeros(par['nm']);
	else:
		m=din;
		if 'd' in par and add==1:
			d=par['d'];
		else:
			d=np.zeros(par['nd']);
# 	print(adj)
# 	print(add)
# 	print(nm,nd)
# 	print('m',m)
# 	print('d',d)
	m,d  = str_adjnull( adj,add,nm,nd,m,d );

	if adj==1:
		m=m+d*w; #dot product
	else: #forward
		d=d+m*w; #d becomes model, m becomes data


	if adj==1:
		dout=m;
	else:
		dout=d;

	return dout
	


def str_trianglen_lop(din,par,adj,add ):
	# str_trianglen_lop: N-D triangle smoothing operator (verified)
	# 
	# Ported to Python by Yangkang Chen, 2022
	# 
	# INPUT
	# din: model/data
	# par: parameter
	# adj: adj flag
	# add: add flag
	# OUTPUT
	# dout: data/model
	import numpy as np
	if adj==1:
		d=din;
		if 'm' in par and add==1:
			m=par['m'];
		else:
			m=np.zeros(par['nm']);
	else:
		m=din;
		if 'd' in par and add==1:
			d=par['d'];
		else:
			d=np.zeros(par['nd']);


	nm=par['nm'];	 #int
	nd=par['nd'];	 #int
	ndim=par['ndim']; #int
	nbox=par['nbox']; #vector[ndim]
	ndat=par['ndat']; #vector[ndim]

	[ m,d ] = str_adjnull( adj,add,nm,nd,m,d );

	tr = [];

	s =[1,ndat[0],ndat[0]*ndat[1]];

	for i in range(0,ndim):
		if (nbox[i] > 1):
			nnp = ndat[i] + 2*nbox[i];
			wt = 1.0 / (nbox[i]*nbox[i]);
			tr.append({'nx':ndat[i], 'nb':nbox[i], 'box':0, 'np':nnp, 'wt':wt, 'tmp':np.zeros(nnp)});
		else:
			tr.append('NULL');

	if adj==1:
		tmp=d;
	else:
		tmp=m;


	for i in range(0,ndim):
		if tr[i] != 'NULL':
			for j in range(0,int(nd/ndat[i])):
				i0=str_first_index(i,j,ndim,ndat,s);
				[tmp,tr[i]]=str_smooth2(tr[i],i0,s[i],0,tmp);

	if adj==1:
		m=m+tmp;
	else:
		d=d+tmp;
		
	if adj==1:
		dout=m;
	else:
		dout=d;

	return dout


def str_first_index( i, j, dim, n, s ):
	#str_first_index: Find first index for multidimensional transforms
	#Ported to Python by Yangkang Chen, 2022
	#
	#INPUT
	#i:	dimension [0...dim-1]
	#j:	line coordinate
	#dim:  number of dimensions
	#n:	box size [dim], vector
	#s:	step [dim], vector
	#OUTPUT
	#i0:   first index

	import numpy as np
	n123 = 1;
	i0 = 0;
	for k in range(0,dim):
		if (k == i):
			continue;
		ii = np.floor(np.mod((j/n123), n[k]));
		n123 = n123 * n[k];
		i0 = i0 + ii * s[k];

	return int(i0)


def str_smooth2( tr, o, d, der, x):
	#str_smooth2: apply triangle smoothing
	#
	#Ported to Python by Yangkang Chen, 2022
	#
	#INPUT
	#tr:   smoothing object
	#o:	trace sampling
	#d:	trace sampling
	#x:	data (smoothed in place)
	#der:  if derivative
	#OUTPUT
	#x: smoothed result
	#tr: triangle struct

	tr['tmp'] = triple2(o, d, tr['nx'], tr['nb'], x, tr['tmp'], tr['box'], tr['wt']);
	tr['tmp'] = doubint2(tr['np'], tr['tmp'], (tr['box'] or der));
	x = fold2(o, d, tr['nx'], tr['nb'], tr['np'], x, tr['tmp']);

	return x,tr


def triple2( o, d, nx, nb, x, tmp, box, wt ):
	#BY Yangkang Chen, Nov, 04, 2019

	for i in range(0,nx+2*nb):
		tmp[i] = 0;

	if box:
		tmp[1:]	 = cblas_saxpy(nx,  +wt,x[o:],d,tmp[1:],   1); 	#y += a*x
		tmp[2*nb:]  = cblas_saxpy(nx,  -wt,x[o:],d,tmp[2*nb:],1);
	else:
# 		print(type(o))
# 		print(o)
		tmp		 = cblas_saxpy(nx,  -wt,x[o:],d,tmp,	   1); 	#y += a*x
		tmp[nb:]	= cblas_saxpy(nx,2.*wt,x[o:],d,tmp[nb:],  1);
		tmp[2*nb:]  = cblas_saxpy(nx,  -wt,x[o:],d,tmp[2*nb:],1);

	return tmp

def doubint2( nx, xx, der ):
	#Modified by Yangkang Chen, Nov, 04, 2019
	#integrate forward
	t = 0.0;
	for i in range(0,nx):
		t = t + xx[i];
		xx[i] = t;

	if der:
		return xx

	#integrate backward
	t = 0.0;
	for i in range(nx-1,-1,-1):
		t = t + xx[i];
		xx[i] = t

	return xx



def cblas_saxpy( n, a, x, sx, y, sy ):
	#y += a*x
	#Modified by Yangkang Chen, Nov, 04, 2019

	for i in range(0,n):
		ix = i * sx;
		iy = i * sy;
		y[iy] = y[iy] + a * x[ix];

	return y

def fold2(o, d, nx, nb, np, x, tmp):
	#Modified by Yangkang Chen, Nov, 04, 2019

	#copy middle
	for i in range(0,nx):
		x[o+i*d] = tmp[i+nb];

	#reflections from the right side
	for j in range(nb+nx,np+1,nx):
		if (nx <= np-j):
			for i in range(0,nx):
				x[o+(nx-1-i)*d] = x[o+(nx-1-i)*d] + tmp[j+i];
		else:
			for i in range(0,np-j):
				x[o+(nx-1-i)*d] = x[o+(nx-1-i)*d] + tmp[j+i];
		j = j + nx;
		if (nx <= np-j):
			for i in range(0,nx): 
				x[o+i*d] = x[o+i*d] + tmp[j+i];
		else:
			for i in range(0,np-j):
				x[o+i*d] = x[o+i*d] + tmp[j+i];

	#reflections from the left side
	for j in range(nb,-1,-nx):
		if (nx <= j):
			for i in range(0,nx):
				x[o+i*d] = x[o+i*d] + tmp[j-1-i];
		else:
			for i in range(0,j):
				x[o+i*d] = x[o+i*d] + tmp[j-1-i];
		j = j - nx;
		if (nx <= j):
			for i in range(0,nx):
				x[o+(nx-1-i)*d] = x[o+(nx-1-i)*d] + tmp[j-1-i];
		else:
			for i in range(0,j):
				x[o+(nx-1-i)*d] = x[o+(nx-1-i)*d] + tmp[j-1-i];
	return x

def str_adjnull( adj,add,nm,nd,m,d ):
	#Claerbout-style adjoint zeroing Zeros out the output (unless add is true). 
	#Useful first step for and linear operator.
	# 
	#  This program is free software; you can redistribute it and/or modify
	#  it under the terms of the GNU General Public License as published by
	#  the Free Software Foundation; either version 2 of the License, or
	#  (at your option) and later version.
	#  
	#  This program is distributed in the hope that it will be useful,
	#  but WITHOUT ANY WARRANTY; without even the implied warranty of
	#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	#  GNU General Public License for more details.
	#  
	#  You should have received a copy of the GNU General Public License
	#  along with this program; if not, write to the Free Software
	#  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
	#adj : adjoint flag; add: addition flag; nm: size of m; nd: size of d
	import numpy as np
	if add:
		return m,d

	if adj:
		m=np.zeros(nm);
		for i in range(0,nm):
			m[i] = 0.0;
	else:
		d=np.zeros(nd);
		for i in range(0,nd):
			d[i] = 0.0;

	return m,d


def str_conjgrad(opP,opL,opS, p, x, dat, eps_cg, tol_cg, N,ifhasp0,par_P,par_L,par_S,verb):
	#str_conjgrad: conjugate gradient with shaping
	#
	#Ported to Python by Yangkang Chen, 2022
	#
	#Modified by Yangkang Chen, Nov, 09, 2019 (fix the "adding" for each oper)
	#
	#INPUT
	#opP: preconditioning operator
	#opL: forward/linear operator
	#opS: shaping operator
	#d:  data
	#N:  number of iterations
	#eps_cg:  scaling
	#tol_cg:  tolerance
	#ifhasp0: flag indicating if has initial model
	#par_P: parameters for P
	#par_L: parameters for L
	#par_S: parameters for S
	#verb: verbosity flag
	#
	#OUPUT
	#x: estimated model
	#
	import numpy as np
	nnp=p.size;
	nx=par_L['nm'];	#model size
	nd=par_L['nd'];	#data size

	if opP != 'NULL':
		d=-dat; #nd*1
		r=opP(d,par_P,0,0);
	else:
		r=-dat;  

	if ifhasp0:
		x=op_S(p,par_S,0,0);
		if opP != 'NULL':
			d=opL(x,par_L,0,0);
			par_P['d']=r;#initialize data
			r=opP(d,par_P,0,1);
		else:
			par_P['d']=r;#initialize data
			r=opL(x,par_L,0,1);

	else:
		p=np.zeros(nnp);#define np!
		x=np.zeros(nx);#define nx!

	dg=0;
	g0=0;
	gnp=0;
	r0=sum(r*r);   #nr*1

	for n in range(1,N+1):
		gp=eps_cg*p; #np*1
		gx=-eps_cg*x; #nx*1
		
		if opP !='NULL':
			d=opP(r,par_P,1,0);#adjoint
			par_L['m']=gx;#initialize model
			gx=opL(d,par_L,1,1);#adjoint,adding
		else:
			par_L['m']=gx;#initialize model
			gx=opL(r,par_L,1,1);#adjoint,adding

	
		par_S['m']=gp;#initialize model
		gp=opS(gx,par_S,1,1);#adjoint,adding
		gx=opS(gp,par_S,0,0);#forward,adding
		
		if opP!='NULL':
			d=opL(gx,par_P,0,0);#forward
			gr=opP(d,par_L,0,0);#forward
		else:
			gr=opL(gx,par_L,0,0);#forward

	
		gn = sum(gp*gp); #np*1
	
		if n==1:
			g0=gn;
			sp=gp; #np*1
			sx=gx; #nx*1
			sr=gr; #nr*1
		else:
			alpha=gn/gnp;
			dg=gn/g0;
		
			if alpha < tol_cg or dg < tol_cg:
				return x;
				break;
		
			gp=alpha*sp+gp;
			t=sp;sp=gp;gp=t;
		
			gx=alpha*sx+gx;
			t=sx;sx=gx;gx=t;
		
			gr=alpha*sr+gr;
			t=sr;sr=gr;gr=t;

	 
		beta=sum(sr*sr)+eps_cg*(sum(sp*sp)-sum(sx*sx));
		
		if verb:
			print('iteration: %d, res: %g !'%(n,sum(r* r) / r0));  

		
		alpha=-gn/beta;
	
		p=alpha*sp+p;
		x=alpha*sx+x;
		r=alpha*sr+r;
	
		gnp=gn;

	return x





	