³Ê
Ç
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718¯	
y
dense_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_0/kernel
r
"dense_0/kernel/Read/ReadVariableOpReadVariableOpdense_0/kernel*
_output_shapes
:	?*
dtype0
q
dense_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_0/bias
j
 dense_0/bias/Read/ReadVariableOpReadVariableOpdense_0/bias*
_output_shapes	
:*
dtype0
y
val_0_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_nameval_0_3/kernel
r
"val_0_3/kernel/Read/ReadVariableOpReadVariableOpval_0_3/kernel*
_output_shapes
:	@*
dtype0
p
val_0_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameval_0_3/bias
i
 val_0_3/bias/Read/ReadVariableOpReadVariableOpval_0_3/bias*
_output_shapes
:@*
dtype0
y
adv_0_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_nameadv_0_3/kernel
r
"adv_0_3/kernel/Read/ReadVariableOpReadVariableOpadv_0_3/kernel*
_output_shapes
:	@*
dtype0
p
adv_0_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameadv_0_3/bias
i
 adv_0_3/bias/Read/ReadVariableOpReadVariableOpadv_0_3/bias*
_output_shapes
:@*
dtype0
x
val_1_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_nameval_1_3/kernel
q
"val_1_3/kernel/Read/ReadVariableOpReadVariableOpval_1_3/kernel*
_output_shapes

:@ *
dtype0
p
val_1_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameval_1_3/bias
i
 val_1_3/bias/Read/ReadVariableOpReadVariableOpval_1_3/bias*
_output_shapes
: *
dtype0
x
adv_1_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_nameadv_1_3/kernel
q
"adv_1_3/kernel/Read/ReadVariableOpReadVariableOpadv_1_3/kernel*
_output_shapes

:@ *
dtype0
p
adv_1_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameadv_1_3/bias
i
 adv_1_3/bias/Read/ReadVariableOpReadVariableOpadv_1_3/bias*
_output_shapes
: *
dtype0
x
val_2_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_nameval_2_3/kernel
q
"val_2_3/kernel/Read/ReadVariableOpReadVariableOpval_2_3/kernel*
_output_shapes

: *
dtype0
p
val_2_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameval_2_3/bias
i
 val_2_3/bias/Read/ReadVariableOpReadVariableOpval_2_3/bias*
_output_shapes
:*
dtype0
x
adv_2_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_nameadv_2_3/kernel
q
"adv_2_3/kernel/Read/ReadVariableOpReadVariableOpadv_2_3/kernel*
_output_shapes

: *
dtype0
p
adv_2_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameadv_2_3/bias
i
 adv_2_3/bias/Read/ReadVariableOpReadVariableOpadv_2_3/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/dense_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_0/kernel/m

)Adam/dense_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_0/kernel/m*
_output_shapes
:	?*
dtype0

Adam/dense_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_0/bias/m
x
'Adam/dense_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_0/bias/m*
_output_shapes	
:*
dtype0

Adam/val_0_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*&
shared_nameAdam/val_0_3/kernel/m

)Adam/val_0_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/val_0_3/kernel/m*
_output_shapes
:	@*
dtype0
~
Adam/val_0_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/val_0_3/bias/m
w
'Adam/val_0_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/val_0_3/bias/m*
_output_shapes
:@*
dtype0

Adam/adv_0_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*&
shared_nameAdam/adv_0_3/kernel/m

)Adam/adv_0_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/adv_0_3/kernel/m*
_output_shapes
:	@*
dtype0
~
Adam/adv_0_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/adv_0_3/bias/m
w
'Adam/adv_0_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/adv_0_3/bias/m*
_output_shapes
:@*
dtype0

Adam/val_1_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *&
shared_nameAdam/val_1_3/kernel/m

)Adam/val_1_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/val_1_3/kernel/m*
_output_shapes

:@ *
dtype0
~
Adam/val_1_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/val_1_3/bias/m
w
'Adam/val_1_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/val_1_3/bias/m*
_output_shapes
: *
dtype0

Adam/adv_1_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *&
shared_nameAdam/adv_1_3/kernel/m

)Adam/adv_1_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/adv_1_3/kernel/m*
_output_shapes

:@ *
dtype0
~
Adam/adv_1_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/adv_1_3/bias/m
w
'Adam/adv_1_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/adv_1_3/bias/m*
_output_shapes
: *
dtype0

Adam/val_2_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/val_2_3/kernel/m

)Adam/val_2_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/val_2_3/kernel/m*
_output_shapes

: *
dtype0
~
Adam/val_2_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/val_2_3/bias/m
w
'Adam/val_2_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/val_2_3/bias/m*
_output_shapes
:*
dtype0

Adam/adv_2_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/adv_2_3/kernel/m

)Adam/adv_2_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/adv_2_3/kernel/m*
_output_shapes

: *
dtype0
~
Adam/adv_2_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/adv_2_3/bias/m
w
'Adam/adv_2_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/adv_2_3/bias/m*
_output_shapes
:*
dtype0

Adam/dense_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_0/kernel/v

)Adam/dense_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_0/kernel/v*
_output_shapes
:	?*
dtype0

Adam/dense_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_0/bias/v
x
'Adam/dense_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_0/bias/v*
_output_shapes	
:*
dtype0

Adam/val_0_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*&
shared_nameAdam/val_0_3/kernel/v

)Adam/val_0_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/val_0_3/kernel/v*
_output_shapes
:	@*
dtype0
~
Adam/val_0_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/val_0_3/bias/v
w
'Adam/val_0_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/val_0_3/bias/v*
_output_shapes
:@*
dtype0

Adam/adv_0_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*&
shared_nameAdam/adv_0_3/kernel/v

)Adam/adv_0_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/adv_0_3/kernel/v*
_output_shapes
:	@*
dtype0
~
Adam/adv_0_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/adv_0_3/bias/v
w
'Adam/adv_0_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/adv_0_3/bias/v*
_output_shapes
:@*
dtype0

Adam/val_1_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *&
shared_nameAdam/val_1_3/kernel/v

)Adam/val_1_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/val_1_3/kernel/v*
_output_shapes

:@ *
dtype0
~
Adam/val_1_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/val_1_3/bias/v
w
'Adam/val_1_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/val_1_3/bias/v*
_output_shapes
: *
dtype0

Adam/adv_1_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *&
shared_nameAdam/adv_1_3/kernel/v

)Adam/adv_1_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/adv_1_3/kernel/v*
_output_shapes

:@ *
dtype0
~
Adam/adv_1_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/adv_1_3/bias/v
w
'Adam/adv_1_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/adv_1_3/bias/v*
_output_shapes
: *
dtype0

Adam/val_2_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/val_2_3/kernel/v

)Adam/val_2_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/val_2_3/kernel/v*
_output_shapes

: *
dtype0
~
Adam/val_2_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/val_2_3/bias/v
w
'Adam/val_2_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/val_2_3/bias/v*
_output_shapes
:*
dtype0

Adam/adv_2_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/adv_2_3/kernel/v

)Adam/adv_2_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/adv_2_3/kernel/v*
_output_shapes

: *
dtype0
~
Adam/adv_2_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/adv_2_3/bias/v
w
'Adam/adv_2_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/adv_2_3/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
G
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ÁF
value·FB´F B­F
·
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer-8

layer-9
layer-10
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
 trainable_variables
!regularization_losses
"	variables
#	keras_api
h

$kernel
%bias
&trainable_variables
'regularization_losses
(	variables
)	keras_api
h

*kernel
+bias
,trainable_variables
-regularization_losses
.	variables
/	keras_api
h

0kernel
1bias
2trainable_variables
3regularization_losses
4	variables
5	keras_api
h

6kernel
7bias
8trainable_variables
9regularization_losses
:	variables
;	keras_api

<	keras_api

=	keras_api

>	keras_api
É
?iter

@beta_1

Abeta_2
	Bdecay
Clearning_ratemqmrmsmtmumv$mw%mx*my+mz0m{1m|6m}7m~vvvvvv$v%v*v+v0v1v6v7v
f
0
1
2
3
4
5
$6
%7
*8
+9
010
111
612
713
 
f
0
1
2
3
4
5
$6
%7
*8
+9
010
111
612
713
­
trainable_variables
regularization_losses
Dnon_trainable_variables
Emetrics

Flayers
Glayer_metrics
	variables
Hlayer_regularization_losses
 
ZX
VARIABLE_VALUEdense_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
trainable_variables
regularization_losses
Inon_trainable_variables
Jmetrics

Klayers
Llayer_metrics
	variables
Mlayer_regularization_losses
ZX
VARIABLE_VALUEval_0_3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEval_0_3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
trainable_variables
regularization_losses
Nnon_trainable_variables
Ometrics

Players
Qlayer_metrics
	variables
Rlayer_regularization_losses
ZX
VARIABLE_VALUEadv_0_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEadv_0_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
 trainable_variables
!regularization_losses
Snon_trainable_variables
Tmetrics

Ulayers
Vlayer_metrics
"	variables
Wlayer_regularization_losses
ZX
VARIABLE_VALUEval_1_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEval_1_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
 

$0
%1
­
&trainable_variables
'regularization_losses
Xnon_trainable_variables
Ymetrics

Zlayers
[layer_metrics
(	variables
\layer_regularization_losses
ZX
VARIABLE_VALUEadv_1_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEadv_1_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1
 

*0
+1
­
,trainable_variables
-regularization_losses
]non_trainable_variables
^metrics

_layers
`layer_metrics
.	variables
alayer_regularization_losses
ZX
VARIABLE_VALUEval_2_3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEval_2_3/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11
 

00
11
­
2trainable_variables
3regularization_losses
bnon_trainable_variables
cmetrics

dlayers
elayer_metrics
4	variables
flayer_regularization_losses
ZX
VARIABLE_VALUEadv_2_3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEadv_2_3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71
 

60
71
­
8trainable_variables
9regularization_losses
gnon_trainable_variables
hmetrics

ilayers
jlayer_metrics
:	variables
klayer_regularization_losses
 
 
 
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

l0
N
0
1
2
3
4
5
6
7
	8

9
10
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	mtotal
	ncount
o	variables
p	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

m0
n1

o	variables
}{
VARIABLE_VALUEAdam/dense_0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/val_0_3/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/val_0_3/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/adv_0_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/adv_0_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/val_1_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/val_1_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/adv_1_3/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/adv_1_3/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/val_2_3/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/val_2_3/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/adv_2_3/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/adv_2_3/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/val_0_3/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/val_0_3/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/adv_0_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/adv_0_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/val_1_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/val_1_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/adv_1_3/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/adv_1_3/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/val_2_3/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/val_2_3/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/adv_2_3/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/adv_2_3/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
serving_default_observationPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ?
ª
StatefulPartitionedCallStatefulPartitionedCallserving_default_observationdense_0/kerneldense_0/biasadv_0_3/kerneladv_0_3/biasval_0_3/kernelval_0_3/biasadv_1_3/kerneladv_1_3/biasval_1_3/kernelval_1_3/biasadv_2_3/kerneladv_2_3/biasval_2_3/kernelval_2_3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *0
f+R)
'__inference_signature_wrapper_263224507
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
z
StaticRegexFullMatchStaticRegexFullMatchsaver_filename"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
\
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
a
Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
h
SelectSelectStaticRegexFullMatchConst_1Const_2"/device:CPU:**
T0*
_output_shapes
: 
`

StringJoin
StringJoinsaver_filenameSelect"/device:CPU:**
N*
_output_shapes
: 
L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
x
ShardedFilenameShardedFilename
StringJoinShardedFilename/shard
num_shards"/device:CPU:0*
_output_shapes
: 
ñ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*
valueB2B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
Ñ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 

SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices"dense_0/kernel/Read/ReadVariableOp dense_0/bias/Read/ReadVariableOp"val_0_3/kernel/Read/ReadVariableOp val_0_3/bias/Read/ReadVariableOp"adv_0_3/kernel/Read/ReadVariableOp adv_0_3/bias/Read/ReadVariableOp"val_1_3/kernel/Read/ReadVariableOp val_1_3/bias/Read/ReadVariableOp"adv_1_3/kernel/Read/ReadVariableOp adv_1_3/bias/Read/ReadVariableOp"val_2_3/kernel/Read/ReadVariableOp val_2_3/bias/Read/ReadVariableOp"adv_2_3/kernel/Read/ReadVariableOp adv_2_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_0/kernel/m/Read/ReadVariableOp'Adam/dense_0/bias/m/Read/ReadVariableOp)Adam/val_0_3/kernel/m/Read/ReadVariableOp'Adam/val_0_3/bias/m/Read/ReadVariableOp)Adam/adv_0_3/kernel/m/Read/ReadVariableOp'Adam/adv_0_3/bias/m/Read/ReadVariableOp)Adam/val_1_3/kernel/m/Read/ReadVariableOp'Adam/val_1_3/bias/m/Read/ReadVariableOp)Adam/adv_1_3/kernel/m/Read/ReadVariableOp'Adam/adv_1_3/bias/m/Read/ReadVariableOp)Adam/val_2_3/kernel/m/Read/ReadVariableOp'Adam/val_2_3/bias/m/Read/ReadVariableOp)Adam/adv_2_3/kernel/m/Read/ReadVariableOp'Adam/adv_2_3/bias/m/Read/ReadVariableOp)Adam/dense_0/kernel/v/Read/ReadVariableOp'Adam/dense_0/bias/v/Read/ReadVariableOp)Adam/val_0_3/kernel/v/Read/ReadVariableOp'Adam/val_0_3/bias/v/Read/ReadVariableOp)Adam/adv_0_3/kernel/v/Read/ReadVariableOp'Adam/adv_0_3/bias/v/Read/ReadVariableOp)Adam/val_1_3/kernel/v/Read/ReadVariableOp'Adam/val_1_3/bias/v/Read/ReadVariableOp)Adam/adv_1_3/kernel/v/Read/ReadVariableOp'Adam/adv_1_3/bias/v/Read/ReadVariableOp)Adam/val_2_3/kernel/v/Read/ReadVariableOp'Adam/val_2_3/bias/v/Read/ReadVariableOp)Adam/adv_2_3/kernel/v/Read/ReadVariableOp'Adam/adv_2_3/bias/v/Read/ReadVariableOpConst"/device:CPU:0*@
dtypes6
422	

&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
o
MergeV2CheckpointsMergeV2Checkpoints&MergeV2Checkpoints/checkpoint_prefixessaver_filename"/device:CPU:0
i
IdentityIdentitysaver_filename^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
ô
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*
valueB2B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
Ô
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 

	RestoreV2	RestoreV2saver_filenameRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*Þ
_output_shapesË
È::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422	
S

Identity_1Identity	RestoreV2"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOpAssignVariableOpdense_0/kernel
Identity_1"/device:CPU:0*
dtype0
U

Identity_2IdentityRestoreV2:1"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_1AssignVariableOpdense_0/bias
Identity_2"/device:CPU:0*
dtype0
U

Identity_3IdentityRestoreV2:2"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_2AssignVariableOpval_0_3/kernel
Identity_3"/device:CPU:0*
dtype0
U

Identity_4IdentityRestoreV2:3"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_3AssignVariableOpval_0_3/bias
Identity_4"/device:CPU:0*
dtype0
U

Identity_5IdentityRestoreV2:4"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_4AssignVariableOpadv_0_3/kernel
Identity_5"/device:CPU:0*
dtype0
U

Identity_6IdentityRestoreV2:5"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_5AssignVariableOpadv_0_3/bias
Identity_6"/device:CPU:0*
dtype0
U

Identity_7IdentityRestoreV2:6"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_6AssignVariableOpval_1_3/kernel
Identity_7"/device:CPU:0*
dtype0
U

Identity_8IdentityRestoreV2:7"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_7AssignVariableOpval_1_3/bias
Identity_8"/device:CPU:0*
dtype0
U

Identity_9IdentityRestoreV2:8"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_8AssignVariableOpadv_1_3/kernel
Identity_9"/device:CPU:0*
dtype0
V
Identity_10IdentityRestoreV2:9"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_9AssignVariableOpadv_1_3/biasIdentity_10"/device:CPU:0*
dtype0
W
Identity_11IdentityRestoreV2:10"/device:CPU:0*
T0*
_output_shapes
:
`
AssignVariableOp_10AssignVariableOpval_2_3/kernelIdentity_11"/device:CPU:0*
dtype0
W
Identity_12IdentityRestoreV2:11"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_11AssignVariableOpval_2_3/biasIdentity_12"/device:CPU:0*
dtype0
W
Identity_13IdentityRestoreV2:12"/device:CPU:0*
T0*
_output_shapes
:
`
AssignVariableOp_12AssignVariableOpadv_2_3/kernelIdentity_13"/device:CPU:0*
dtype0
W
Identity_14IdentityRestoreV2:13"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_13AssignVariableOpadv_2_3/biasIdentity_14"/device:CPU:0*
dtype0
W
Identity_15IdentityRestoreV2:14"/device:CPU:0*
T0	*
_output_shapes
:
[
AssignVariableOp_14AssignVariableOp	Adam/iterIdentity_15"/device:CPU:0*
dtype0	
W
Identity_16IdentityRestoreV2:15"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_15AssignVariableOpAdam/beta_1Identity_16"/device:CPU:0*
dtype0
W
Identity_17IdentityRestoreV2:16"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_16AssignVariableOpAdam/beta_2Identity_17"/device:CPU:0*
dtype0
W
Identity_18IdentityRestoreV2:17"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_17AssignVariableOp
Adam/decayIdentity_18"/device:CPU:0*
dtype0
W
Identity_19IdentityRestoreV2:18"/device:CPU:0*
T0*
_output_shapes
:
d
AssignVariableOp_18AssignVariableOpAdam/learning_rateIdentity_19"/device:CPU:0*
dtype0
W
Identity_20IdentityRestoreV2:19"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_19AssignVariableOptotalIdentity_20"/device:CPU:0*
dtype0
W
Identity_21IdentityRestoreV2:20"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_20AssignVariableOpcountIdentity_21"/device:CPU:0*
dtype0
W
Identity_22IdentityRestoreV2:21"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_21AssignVariableOpAdam/dense_0/kernel/mIdentity_22"/device:CPU:0*
dtype0
W
Identity_23IdentityRestoreV2:22"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_22AssignVariableOpAdam/dense_0/bias/mIdentity_23"/device:CPU:0*
dtype0
W
Identity_24IdentityRestoreV2:23"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_23AssignVariableOpAdam/val_0_3/kernel/mIdentity_24"/device:CPU:0*
dtype0
W
Identity_25IdentityRestoreV2:24"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_24AssignVariableOpAdam/val_0_3/bias/mIdentity_25"/device:CPU:0*
dtype0
W
Identity_26IdentityRestoreV2:25"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_25AssignVariableOpAdam/adv_0_3/kernel/mIdentity_26"/device:CPU:0*
dtype0
W
Identity_27IdentityRestoreV2:26"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_26AssignVariableOpAdam/adv_0_3/bias/mIdentity_27"/device:CPU:0*
dtype0
W
Identity_28IdentityRestoreV2:27"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_27AssignVariableOpAdam/val_1_3/kernel/mIdentity_28"/device:CPU:0*
dtype0
W
Identity_29IdentityRestoreV2:28"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_28AssignVariableOpAdam/val_1_3/bias/mIdentity_29"/device:CPU:0*
dtype0
W
Identity_30IdentityRestoreV2:29"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_29AssignVariableOpAdam/adv_1_3/kernel/mIdentity_30"/device:CPU:0*
dtype0
W
Identity_31IdentityRestoreV2:30"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_30AssignVariableOpAdam/adv_1_3/bias/mIdentity_31"/device:CPU:0*
dtype0
W
Identity_32IdentityRestoreV2:31"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_31AssignVariableOpAdam/val_2_3/kernel/mIdentity_32"/device:CPU:0*
dtype0
W
Identity_33IdentityRestoreV2:32"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_32AssignVariableOpAdam/val_2_3/bias/mIdentity_33"/device:CPU:0*
dtype0
W
Identity_34IdentityRestoreV2:33"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_33AssignVariableOpAdam/adv_2_3/kernel/mIdentity_34"/device:CPU:0*
dtype0
W
Identity_35IdentityRestoreV2:34"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_34AssignVariableOpAdam/adv_2_3/bias/mIdentity_35"/device:CPU:0*
dtype0
W
Identity_36IdentityRestoreV2:35"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_35AssignVariableOpAdam/dense_0/kernel/vIdentity_36"/device:CPU:0*
dtype0
W
Identity_37IdentityRestoreV2:36"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_36AssignVariableOpAdam/dense_0/bias/vIdentity_37"/device:CPU:0*
dtype0
W
Identity_38IdentityRestoreV2:37"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_37AssignVariableOpAdam/val_0_3/kernel/vIdentity_38"/device:CPU:0*
dtype0
W
Identity_39IdentityRestoreV2:38"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_38AssignVariableOpAdam/val_0_3/bias/vIdentity_39"/device:CPU:0*
dtype0
W
Identity_40IdentityRestoreV2:39"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_39AssignVariableOpAdam/adv_0_3/kernel/vIdentity_40"/device:CPU:0*
dtype0
W
Identity_41IdentityRestoreV2:40"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_40AssignVariableOpAdam/adv_0_3/bias/vIdentity_41"/device:CPU:0*
dtype0
W
Identity_42IdentityRestoreV2:41"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_41AssignVariableOpAdam/val_1_3/kernel/vIdentity_42"/device:CPU:0*
dtype0
W
Identity_43IdentityRestoreV2:42"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_42AssignVariableOpAdam/val_1_3/bias/vIdentity_43"/device:CPU:0*
dtype0
W
Identity_44IdentityRestoreV2:43"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_43AssignVariableOpAdam/adv_1_3/kernel/vIdentity_44"/device:CPU:0*
dtype0
W
Identity_45IdentityRestoreV2:44"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_44AssignVariableOpAdam/adv_1_3/bias/vIdentity_45"/device:CPU:0*
dtype0
W
Identity_46IdentityRestoreV2:45"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_45AssignVariableOpAdam/val_2_3/kernel/vIdentity_46"/device:CPU:0*
dtype0
W
Identity_47IdentityRestoreV2:46"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_46AssignVariableOpAdam/val_2_3/bias/vIdentity_47"/device:CPU:0*
dtype0
W
Identity_48IdentityRestoreV2:47"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_47AssignVariableOpAdam/adv_2_3/kernel/vIdentity_48"/device:CPU:0*
dtype0
W
Identity_49IdentityRestoreV2:48"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_48AssignVariableOpAdam/adv_2_3/bias/vIdentity_49"/device:CPU:0*
dtype0

NoOp_1NoOp"/device:CPU:0
	
Identity_50Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp_1"/device:CPU:0*
T0*
_output_shapes
: ¾
Ò	
÷
F__inference_val_2_3_layer_call_and_return_conditional_losses_263224847

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
·	
Ü
+__inference_val_2_3_layer_call_fn_263224857

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
®

÷
F__inference_adv_1_3_layer_call_and_return_conditional_losses_263224826

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
F
Ç

A__inference_bootstrapped_ddqn_head_3_of_4_layer_call_fn_263224356
observation9
&dense_0_matmul_readvariableop_resource:	?6
'dense_0_biasadd_readvariableop_resource:	9
&adv_0_3_matmul_readvariableop_resource:	@5
'adv_0_3_biasadd_readvariableop_resource:@9
&val_0_3_matmul_readvariableop_resource:	@5
'val_0_3_biasadd_readvariableop_resource:@8
&adv_1_3_matmul_readvariableop_resource:@ 5
'adv_1_3_biasadd_readvariableop_resource: 8
&val_1_3_matmul_readvariableop_resource:@ 5
'val_1_3_biasadd_readvariableop_resource: 8
&adv_2_3_matmul_readvariableop_resource: 5
'adv_2_3_biasadd_readvariableop_resource:8
&val_2_3_matmul_readvariableop_resource: 5
'val_2_3_biasadd_readvariableop_resource:
identity¢adv_0_3/BiasAdd/ReadVariableOp¢adv_0_3/MatMul/ReadVariableOp¢adv_1_3/BiasAdd/ReadVariableOp¢adv_1_3/MatMul/ReadVariableOp¢adv_2_3/BiasAdd/ReadVariableOp¢adv_2_3/MatMul/ReadVariableOp¢dense_0/BiasAdd/ReadVariableOp¢dense_0/MatMul/ReadVariableOp¢val_0_3/BiasAdd/ReadVariableOp¢val_0_3/MatMul/ReadVariableOp¢val_1_3/BiasAdd/ReadVariableOp¢val_1_3/MatMul/ReadVariableOp¢val_2_3/BiasAdd/ReadVariableOp¢val_2_3/MatMul/ReadVariableOp¦
dense_0/MatMul/ReadVariableOpReadVariableOp&dense_0_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_0/MatMul/ReadVariableOp
dense_0/MatMulMatMulobservation%dense_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_0/MatMul¥
dense_0/BiasAdd/ReadVariableOpReadVariableOp'dense_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_0/BiasAdd/ReadVariableOp¢
dense_0/BiasAddBiasAdddense_0/MatMul:product:0&dense_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_0/BiasAddq
dense_0/ReluReludense_0/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_0/Relu¦
adv_0_3/MatMul/ReadVariableOpReadVariableOp&adv_0_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
adv_0_3/MatMul/ReadVariableOp
adv_0_3/MatMulMatMuldense_0/Relu:activations:0%adv_0_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_0_3/MatMul¤
adv_0_3/BiasAdd/ReadVariableOpReadVariableOp'adv_0_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
adv_0_3/BiasAdd/ReadVariableOp¡
adv_0_3/BiasAddBiasAddadv_0_3/MatMul:product:0&adv_0_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_0_3/BiasAddp
adv_0_3/ReluReluadv_0_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_0_3/Relu¦
val_0_3/MatMul/ReadVariableOpReadVariableOp&val_0_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
val_0_3/MatMul/ReadVariableOp
val_0_3/MatMulMatMuldense_0/Relu:activations:0%val_0_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_0_3/MatMul¤
val_0_3/BiasAdd/ReadVariableOpReadVariableOp'val_0_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
val_0_3/BiasAdd/ReadVariableOp¡
val_0_3/BiasAddBiasAddval_0_3/MatMul:product:0&val_0_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_0_3/BiasAddp
val_0_3/ReluReluval_0_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_0_3/Relu¥
adv_1_3/MatMul/ReadVariableOpReadVariableOp&adv_1_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
adv_1_3/MatMul/ReadVariableOp
adv_1_3/MatMulMatMuladv_0_3/Relu:activations:0%adv_1_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
adv_1_3/MatMul¤
adv_1_3/BiasAdd/ReadVariableOpReadVariableOp'adv_1_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
adv_1_3/BiasAdd/ReadVariableOp¡
adv_1_3/BiasAddBiasAddadv_1_3/MatMul:product:0&adv_1_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
adv_1_3/BiasAddp
adv_1_3/ReluReluadv_1_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
adv_1_3/Relu¥
val_1_3/MatMul/ReadVariableOpReadVariableOp&val_1_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
val_1_3/MatMul/ReadVariableOp
val_1_3/MatMulMatMulval_0_3/Relu:activations:0%val_1_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
val_1_3/MatMul¤
val_1_3/BiasAdd/ReadVariableOpReadVariableOp'val_1_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
val_1_3/BiasAdd/ReadVariableOp¡
val_1_3/BiasAddBiasAddval_1_3/MatMul:product:0&val_1_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
val_1_3/BiasAddp
val_1_3/ReluReluval_1_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
val_1_3/Relu¥
adv_2_3/MatMul/ReadVariableOpReadVariableOp&adv_2_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
adv_2_3/MatMul/ReadVariableOp
adv_2_3/MatMulMatMuladv_1_3/Relu:activations:0%adv_2_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adv_2_3/MatMul¤
adv_2_3/BiasAdd/ReadVariableOpReadVariableOp'adv_2_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
adv_2_3/BiasAdd/ReadVariableOp¡
adv_2_3/BiasAddBiasAddadv_2_3/MatMul:product:0&adv_2_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adv_2_3/BiasAdd¥
val_2_3/MatMul/ReadVariableOpReadVariableOp&val_2_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
val_2_3/MatMul/ReadVariableOp
val_2_3/MatMulMatMulval_1_3/Relu:activations:0%val_2_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
val_2_3/MatMul¤
val_2_3/BiasAdd/ReadVariableOpReadVariableOp'val_2_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
val_2_3/BiasAdd/ReadVariableOp¡
val_2_3/BiasAddBiasAddval_2_3/MatMul:product:0&val_2_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
val_2_3/BiasAdd«
tf.__operators__.add_3/AddV2AddV2val_2_3/BiasAdd:output:0adv_2_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_3/AddV2
,tf.math.reduce_mean_3/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,tf.math.reduce_mean_3/Mean/reduction_indicesÔ
tf.math.reduce_mean_3/MeanMeanadv_2_3/BiasAdd:output:05tf.math.reduce_mean_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
tf.math.reduce_mean_3/Mean°
tf.math.subtract_3/SubSub tf.__operators__.add_3/AddV2:z:0#tf.math.reduce_mean_3/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.subtract_3/Subµ
IdentityIdentitytf.math.subtract_3/Sub:z:0^adv_0_3/BiasAdd/ReadVariableOp^adv_0_3/MatMul/ReadVariableOp^adv_1_3/BiasAdd/ReadVariableOp^adv_1_3/MatMul/ReadVariableOp^adv_2_3/BiasAdd/ReadVariableOp^adv_2_3/MatMul/ReadVariableOp^dense_0/BiasAdd/ReadVariableOp^dense_0/MatMul/ReadVariableOp^val_0_3/BiasAdd/ReadVariableOp^val_0_3/MatMul/ReadVariableOp^val_1_3/BiasAdd/ReadVariableOp^val_1_3/MatMul/ReadVariableOp^val_2_3/BiasAdd/ReadVariableOp^val_2_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ?: : : : : : : : : : : : : : 2@
adv_0_3/BiasAdd/ReadVariableOpadv_0_3/BiasAdd/ReadVariableOp2>
adv_0_3/MatMul/ReadVariableOpadv_0_3/MatMul/ReadVariableOp2@
adv_1_3/BiasAdd/ReadVariableOpadv_1_3/BiasAdd/ReadVariableOp2>
adv_1_3/MatMul/ReadVariableOpadv_1_3/MatMul/ReadVariableOp2@
adv_2_3/BiasAdd/ReadVariableOpadv_2_3/BiasAdd/ReadVariableOp2>
adv_2_3/MatMul/ReadVariableOpadv_2_3/MatMul/ReadVariableOp2@
dense_0/BiasAdd/ReadVariableOpdense_0/BiasAdd/ReadVariableOp2>
dense_0/MatMul/ReadVariableOpdense_0/MatMul/ReadVariableOp2@
val_0_3/BiasAdd/ReadVariableOpval_0_3/BiasAdd/ReadVariableOp2>
val_0_3/MatMul/ReadVariableOpval_0_3/MatMul/ReadVariableOp2@
val_1_3/BiasAdd/ReadVariableOpval_1_3/BiasAdd/ReadVariableOp2>
val_1_3/MatMul/ReadVariableOpval_1_3/MatMul/ReadVariableOp2@
val_2_3/BiasAdd/ReadVariableOpval_2_3/BiasAdd/ReadVariableOp2>
val_2_3/MatMul/ReadVariableOpval_2_3/MatMul/ReadVariableOp:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?
%
_user_specified_nameobservation
F
Ý

\__inference_bootstrapped_ddqn_head_3_of_4_layer_call_and_return_conditional_losses_263224617

inputs9
&dense_0_matmul_readvariableop_resource:	?6
'dense_0_biasadd_readvariableop_resource:	9
&adv_0_3_matmul_readvariableop_resource:	@5
'adv_0_3_biasadd_readvariableop_resource:@9
&val_0_3_matmul_readvariableop_resource:	@5
'val_0_3_biasadd_readvariableop_resource:@8
&adv_1_3_matmul_readvariableop_resource:@ 5
'adv_1_3_biasadd_readvariableop_resource: 8
&val_1_3_matmul_readvariableop_resource:@ 5
'val_1_3_biasadd_readvariableop_resource: 8
&adv_2_3_matmul_readvariableop_resource: 5
'adv_2_3_biasadd_readvariableop_resource:8
&val_2_3_matmul_readvariableop_resource: 5
'val_2_3_biasadd_readvariableop_resource:
identity¢adv_0_3/BiasAdd/ReadVariableOp¢adv_0_3/MatMul/ReadVariableOp¢adv_1_3/BiasAdd/ReadVariableOp¢adv_1_3/MatMul/ReadVariableOp¢adv_2_3/BiasAdd/ReadVariableOp¢adv_2_3/MatMul/ReadVariableOp¢dense_0/BiasAdd/ReadVariableOp¢dense_0/MatMul/ReadVariableOp¢val_0_3/BiasAdd/ReadVariableOp¢val_0_3/MatMul/ReadVariableOp¢val_1_3/BiasAdd/ReadVariableOp¢val_1_3/MatMul/ReadVariableOp¢val_2_3/BiasAdd/ReadVariableOp¢val_2_3/MatMul/ReadVariableOp¦
dense_0/MatMul/ReadVariableOpReadVariableOp&dense_0_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_0/MatMul/ReadVariableOp
dense_0/MatMulMatMulinputs%dense_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_0/MatMul¥
dense_0/BiasAdd/ReadVariableOpReadVariableOp'dense_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_0/BiasAdd/ReadVariableOp¢
dense_0/BiasAddBiasAdddense_0/MatMul:product:0&dense_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_0/BiasAddq
dense_0/ReluReludense_0/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_0/Relu¦
adv_0_3/MatMul/ReadVariableOpReadVariableOp&adv_0_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
adv_0_3/MatMul/ReadVariableOp
adv_0_3/MatMulMatMuldense_0/Relu:activations:0%adv_0_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_0_3/MatMul¤
adv_0_3/BiasAdd/ReadVariableOpReadVariableOp'adv_0_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
adv_0_3/BiasAdd/ReadVariableOp¡
adv_0_3/BiasAddBiasAddadv_0_3/MatMul:product:0&adv_0_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_0_3/BiasAddp
adv_0_3/ReluReluadv_0_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_0_3/Relu¦
val_0_3/MatMul/ReadVariableOpReadVariableOp&val_0_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
val_0_3/MatMul/ReadVariableOp
val_0_3/MatMulMatMuldense_0/Relu:activations:0%val_0_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_0_3/MatMul¤
val_0_3/BiasAdd/ReadVariableOpReadVariableOp'val_0_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
val_0_3/BiasAdd/ReadVariableOp¡
val_0_3/BiasAddBiasAddval_0_3/MatMul:product:0&val_0_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_0_3/BiasAddp
val_0_3/ReluReluval_0_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_0_3/Relu¥
adv_1_3/MatMul/ReadVariableOpReadVariableOp&adv_1_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
adv_1_3/MatMul/ReadVariableOp
adv_1_3/MatMulMatMuladv_0_3/Relu:activations:0%adv_1_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
adv_1_3/MatMul¤
adv_1_3/BiasAdd/ReadVariableOpReadVariableOp'adv_1_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
adv_1_3/BiasAdd/ReadVariableOp¡
adv_1_3/BiasAddBiasAddadv_1_3/MatMul:product:0&adv_1_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
adv_1_3/BiasAddp
adv_1_3/ReluReluadv_1_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
adv_1_3/Relu¥
val_1_3/MatMul/ReadVariableOpReadVariableOp&val_1_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
val_1_3/MatMul/ReadVariableOp
val_1_3/MatMulMatMulval_0_3/Relu:activations:0%val_1_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
val_1_3/MatMul¤
val_1_3/BiasAdd/ReadVariableOpReadVariableOp'val_1_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
val_1_3/BiasAdd/ReadVariableOp¡
val_1_3/BiasAddBiasAddval_1_3/MatMul:product:0&val_1_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
val_1_3/BiasAddp
val_1_3/ReluReluval_1_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
val_1_3/Relu¥
adv_2_3/MatMul/ReadVariableOpReadVariableOp&adv_2_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
adv_2_3/MatMul/ReadVariableOp
adv_2_3/MatMulMatMuladv_1_3/Relu:activations:0%adv_2_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adv_2_3/MatMul¤
adv_2_3/BiasAdd/ReadVariableOpReadVariableOp'adv_2_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
adv_2_3/BiasAdd/ReadVariableOp¡
adv_2_3/BiasAddBiasAddadv_2_3/MatMul:product:0&adv_2_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adv_2_3/BiasAdd¥
val_2_3/MatMul/ReadVariableOpReadVariableOp&val_2_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
val_2_3/MatMul/ReadVariableOp
val_2_3/MatMulMatMulval_1_3/Relu:activations:0%val_2_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
val_2_3/MatMul¤
val_2_3/BiasAdd/ReadVariableOpReadVariableOp'val_2_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
val_2_3/BiasAdd/ReadVariableOp¡
val_2_3/BiasAddBiasAddval_2_3/MatMul:product:0&val_2_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
val_2_3/BiasAdd«
tf.__operators__.add_3/AddV2AddV2val_2_3/BiasAdd:output:0adv_2_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_3/AddV2
,tf.math.reduce_mean_3/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,tf.math.reduce_mean_3/Mean/reduction_indicesÔ
tf.math.reduce_mean_3/MeanMeanadv_2_3/BiasAdd:output:05tf.math.reduce_mean_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
tf.math.reduce_mean_3/Mean°
tf.math.subtract_3/SubSub tf.__operators__.add_3/AddV2:z:0#tf.math.reduce_mean_3/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.subtract_3/Subµ
IdentityIdentitytf.math.subtract_3/Sub:z:0^adv_0_3/BiasAdd/ReadVariableOp^adv_0_3/MatMul/ReadVariableOp^adv_1_3/BiasAdd/ReadVariableOp^adv_1_3/MatMul/ReadVariableOp^adv_2_3/BiasAdd/ReadVariableOp^adv_2_3/MatMul/ReadVariableOp^dense_0/BiasAdd/ReadVariableOp^dense_0/MatMul/ReadVariableOp^val_0_3/BiasAdd/ReadVariableOp^val_0_3/MatMul/ReadVariableOp^val_1_3/BiasAdd/ReadVariableOp^val_1_3/MatMul/ReadVariableOp^val_2_3/BiasAdd/ReadVariableOp^val_2_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ?: : : : : : : : : : : : : : 2@
adv_0_3/BiasAdd/ReadVariableOpadv_0_3/BiasAdd/ReadVariableOp2>
adv_0_3/MatMul/ReadVariableOpadv_0_3/MatMul/ReadVariableOp2@
adv_1_3/BiasAdd/ReadVariableOpadv_1_3/BiasAdd/ReadVariableOp2>
adv_1_3/MatMul/ReadVariableOpadv_1_3/MatMul/ReadVariableOp2@
adv_2_3/BiasAdd/ReadVariableOpadv_2_3/BiasAdd/ReadVariableOp2>
adv_2_3/MatMul/ReadVariableOpadv_2_3/MatMul/ReadVariableOp2@
dense_0/BiasAdd/ReadVariableOpdense_0/BiasAdd/ReadVariableOp2>
dense_0/MatMul/ReadVariableOpdense_0/MatMul/ReadVariableOp2@
val_0_3/BiasAdd/ReadVariableOpval_0_3/BiasAdd/ReadVariableOp2>
val_0_3/MatMul/ReadVariableOpval_0_3/MatMul/ReadVariableOp2@
val_1_3/BiasAdd/ReadVariableOpval_1_3/BiasAdd/ReadVariableOp2>
val_1_3/MatMul/ReadVariableOpval_1_3/MatMul/ReadVariableOp2@
val_2_3/BiasAdd/ReadVariableOpval_2_3/BiasAdd/ReadVariableOp2>
val_2_3/MatMul/ReadVariableOpval_2_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?
 
_user_specified_nameinputs


Ý
+__inference_val_0_3_layer_call_fn_263224771

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·	
Ü
+__inference_adv_2_3_layer_call_fn_263224877

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
t
ò
$__inference__wrapped_model_263223860
observationW
Dbootstrapped_ddqn_head_3_of_4_dense_0_matmul_readvariableop_resource:	?T
Ebootstrapped_ddqn_head_3_of_4_dense_0_biasadd_readvariableop_resource:	W
Dbootstrapped_ddqn_head_3_of_4_adv_0_3_matmul_readvariableop_resource:	@S
Ebootstrapped_ddqn_head_3_of_4_adv_0_3_biasadd_readvariableop_resource:@W
Dbootstrapped_ddqn_head_3_of_4_val_0_3_matmul_readvariableop_resource:	@S
Ebootstrapped_ddqn_head_3_of_4_val_0_3_biasadd_readvariableop_resource:@V
Dbootstrapped_ddqn_head_3_of_4_adv_1_3_matmul_readvariableop_resource:@ S
Ebootstrapped_ddqn_head_3_of_4_adv_1_3_biasadd_readvariableop_resource: V
Dbootstrapped_ddqn_head_3_of_4_val_1_3_matmul_readvariableop_resource:@ S
Ebootstrapped_ddqn_head_3_of_4_val_1_3_biasadd_readvariableop_resource: V
Dbootstrapped_ddqn_head_3_of_4_adv_2_3_matmul_readvariableop_resource: S
Ebootstrapped_ddqn_head_3_of_4_adv_2_3_biasadd_readvariableop_resource:V
Dbootstrapped_ddqn_head_3_of_4_val_2_3_matmul_readvariableop_resource: S
Ebootstrapped_ddqn_head_3_of_4_val_2_3_biasadd_readvariableop_resource:
identity¢<bootstrapped_ddqn_head_3_of_4/adv_0_3/BiasAdd/ReadVariableOp¢;bootstrapped_ddqn_head_3_of_4/adv_0_3/MatMul/ReadVariableOp¢<bootstrapped_ddqn_head_3_of_4/adv_1_3/BiasAdd/ReadVariableOp¢;bootstrapped_ddqn_head_3_of_4/adv_1_3/MatMul/ReadVariableOp¢<bootstrapped_ddqn_head_3_of_4/adv_2_3/BiasAdd/ReadVariableOp¢;bootstrapped_ddqn_head_3_of_4/adv_2_3/MatMul/ReadVariableOp¢<bootstrapped_ddqn_head_3_of_4/dense_0/BiasAdd/ReadVariableOp¢;bootstrapped_ddqn_head_3_of_4/dense_0/MatMul/ReadVariableOp¢<bootstrapped_ddqn_head_3_of_4/val_0_3/BiasAdd/ReadVariableOp¢;bootstrapped_ddqn_head_3_of_4/val_0_3/MatMul/ReadVariableOp¢<bootstrapped_ddqn_head_3_of_4/val_1_3/BiasAdd/ReadVariableOp¢;bootstrapped_ddqn_head_3_of_4/val_1_3/MatMul/ReadVariableOp¢<bootstrapped_ddqn_head_3_of_4/val_2_3/BiasAdd/ReadVariableOp¢;bootstrapped_ddqn_head_3_of_4/val_2_3/MatMul/ReadVariableOp
;bootstrapped_ddqn_head_3_of_4/dense_0/MatMul/ReadVariableOpReadVariableOpDbootstrapped_ddqn_head_3_of_4_dense_0_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02=
;bootstrapped_ddqn_head_3_of_4/dense_0/MatMul/ReadVariableOpë
,bootstrapped_ddqn_head_3_of_4/dense_0/MatMulMatMulobservationCbootstrapped_ddqn_head_3_of_4/dense_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,bootstrapped_ddqn_head_3_of_4/dense_0/MatMulÿ
<bootstrapped_ddqn_head_3_of_4/dense_0/BiasAdd/ReadVariableOpReadVariableOpEbootstrapped_ddqn_head_3_of_4_dense_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02>
<bootstrapped_ddqn_head_3_of_4/dense_0/BiasAdd/ReadVariableOp
-bootstrapped_ddqn_head_3_of_4/dense_0/BiasAddBiasAdd6bootstrapped_ddqn_head_3_of_4/dense_0/MatMul:product:0Dbootstrapped_ddqn_head_3_of_4/dense_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-bootstrapped_ddqn_head_3_of_4/dense_0/BiasAddË
*bootstrapped_ddqn_head_3_of_4/dense_0/ReluRelu6bootstrapped_ddqn_head_3_of_4/dense_0/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*bootstrapped_ddqn_head_3_of_4/dense_0/Relu
;bootstrapped_ddqn_head_3_of_4/adv_0_3/MatMul/ReadVariableOpReadVariableOpDbootstrapped_ddqn_head_3_of_4_adv_0_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02=
;bootstrapped_ddqn_head_3_of_4/adv_0_3/MatMul/ReadVariableOp
,bootstrapped_ddqn_head_3_of_4/adv_0_3/MatMulMatMul8bootstrapped_ddqn_head_3_of_4/dense_0/Relu:activations:0Cbootstrapped_ddqn_head_3_of_4/adv_0_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2.
,bootstrapped_ddqn_head_3_of_4/adv_0_3/MatMulþ
<bootstrapped_ddqn_head_3_of_4/adv_0_3/BiasAdd/ReadVariableOpReadVariableOpEbootstrapped_ddqn_head_3_of_4_adv_0_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02>
<bootstrapped_ddqn_head_3_of_4/adv_0_3/BiasAdd/ReadVariableOp
-bootstrapped_ddqn_head_3_of_4/adv_0_3/BiasAddBiasAdd6bootstrapped_ddqn_head_3_of_4/adv_0_3/MatMul:product:0Dbootstrapped_ddqn_head_3_of_4/adv_0_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2/
-bootstrapped_ddqn_head_3_of_4/adv_0_3/BiasAddÊ
*bootstrapped_ddqn_head_3_of_4/adv_0_3/ReluRelu6bootstrapped_ddqn_head_3_of_4/adv_0_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*bootstrapped_ddqn_head_3_of_4/adv_0_3/Relu
;bootstrapped_ddqn_head_3_of_4/val_0_3/MatMul/ReadVariableOpReadVariableOpDbootstrapped_ddqn_head_3_of_4_val_0_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02=
;bootstrapped_ddqn_head_3_of_4/val_0_3/MatMul/ReadVariableOp
,bootstrapped_ddqn_head_3_of_4/val_0_3/MatMulMatMul8bootstrapped_ddqn_head_3_of_4/dense_0/Relu:activations:0Cbootstrapped_ddqn_head_3_of_4/val_0_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2.
,bootstrapped_ddqn_head_3_of_4/val_0_3/MatMulþ
<bootstrapped_ddqn_head_3_of_4/val_0_3/BiasAdd/ReadVariableOpReadVariableOpEbootstrapped_ddqn_head_3_of_4_val_0_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02>
<bootstrapped_ddqn_head_3_of_4/val_0_3/BiasAdd/ReadVariableOp
-bootstrapped_ddqn_head_3_of_4/val_0_3/BiasAddBiasAdd6bootstrapped_ddqn_head_3_of_4/val_0_3/MatMul:product:0Dbootstrapped_ddqn_head_3_of_4/val_0_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2/
-bootstrapped_ddqn_head_3_of_4/val_0_3/BiasAddÊ
*bootstrapped_ddqn_head_3_of_4/val_0_3/ReluRelu6bootstrapped_ddqn_head_3_of_4/val_0_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*bootstrapped_ddqn_head_3_of_4/val_0_3/Reluÿ
;bootstrapped_ddqn_head_3_of_4/adv_1_3/MatMul/ReadVariableOpReadVariableOpDbootstrapped_ddqn_head_3_of_4_adv_1_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02=
;bootstrapped_ddqn_head_3_of_4/adv_1_3/MatMul/ReadVariableOp
,bootstrapped_ddqn_head_3_of_4/adv_1_3/MatMulMatMul8bootstrapped_ddqn_head_3_of_4/adv_0_3/Relu:activations:0Cbootstrapped_ddqn_head_3_of_4/adv_1_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,bootstrapped_ddqn_head_3_of_4/adv_1_3/MatMulþ
<bootstrapped_ddqn_head_3_of_4/adv_1_3/BiasAdd/ReadVariableOpReadVariableOpEbootstrapped_ddqn_head_3_of_4_adv_1_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02>
<bootstrapped_ddqn_head_3_of_4/adv_1_3/BiasAdd/ReadVariableOp
-bootstrapped_ddqn_head_3_of_4/adv_1_3/BiasAddBiasAdd6bootstrapped_ddqn_head_3_of_4/adv_1_3/MatMul:product:0Dbootstrapped_ddqn_head_3_of_4/adv_1_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-bootstrapped_ddqn_head_3_of_4/adv_1_3/BiasAddÊ
*bootstrapped_ddqn_head_3_of_4/adv_1_3/ReluRelu6bootstrapped_ddqn_head_3_of_4/adv_1_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*bootstrapped_ddqn_head_3_of_4/adv_1_3/Reluÿ
;bootstrapped_ddqn_head_3_of_4/val_1_3/MatMul/ReadVariableOpReadVariableOpDbootstrapped_ddqn_head_3_of_4_val_1_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02=
;bootstrapped_ddqn_head_3_of_4/val_1_3/MatMul/ReadVariableOp
,bootstrapped_ddqn_head_3_of_4/val_1_3/MatMulMatMul8bootstrapped_ddqn_head_3_of_4/val_0_3/Relu:activations:0Cbootstrapped_ddqn_head_3_of_4/val_1_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,bootstrapped_ddqn_head_3_of_4/val_1_3/MatMulþ
<bootstrapped_ddqn_head_3_of_4/val_1_3/BiasAdd/ReadVariableOpReadVariableOpEbootstrapped_ddqn_head_3_of_4_val_1_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02>
<bootstrapped_ddqn_head_3_of_4/val_1_3/BiasAdd/ReadVariableOp
-bootstrapped_ddqn_head_3_of_4/val_1_3/BiasAddBiasAdd6bootstrapped_ddqn_head_3_of_4/val_1_3/MatMul:product:0Dbootstrapped_ddqn_head_3_of_4/val_1_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-bootstrapped_ddqn_head_3_of_4/val_1_3/BiasAddÊ
*bootstrapped_ddqn_head_3_of_4/val_1_3/ReluRelu6bootstrapped_ddqn_head_3_of_4/val_1_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*bootstrapped_ddqn_head_3_of_4/val_1_3/Reluÿ
;bootstrapped_ddqn_head_3_of_4/adv_2_3/MatMul/ReadVariableOpReadVariableOpDbootstrapped_ddqn_head_3_of_4_adv_2_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02=
;bootstrapped_ddqn_head_3_of_4/adv_2_3/MatMul/ReadVariableOp
,bootstrapped_ddqn_head_3_of_4/adv_2_3/MatMulMatMul8bootstrapped_ddqn_head_3_of_4/adv_1_3/Relu:activations:0Cbootstrapped_ddqn_head_3_of_4/adv_2_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,bootstrapped_ddqn_head_3_of_4/adv_2_3/MatMulþ
<bootstrapped_ddqn_head_3_of_4/adv_2_3/BiasAdd/ReadVariableOpReadVariableOpEbootstrapped_ddqn_head_3_of_4_adv_2_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02>
<bootstrapped_ddqn_head_3_of_4/adv_2_3/BiasAdd/ReadVariableOp
-bootstrapped_ddqn_head_3_of_4/adv_2_3/BiasAddBiasAdd6bootstrapped_ddqn_head_3_of_4/adv_2_3/MatMul:product:0Dbootstrapped_ddqn_head_3_of_4/adv_2_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-bootstrapped_ddqn_head_3_of_4/adv_2_3/BiasAddÿ
;bootstrapped_ddqn_head_3_of_4/val_2_3/MatMul/ReadVariableOpReadVariableOpDbootstrapped_ddqn_head_3_of_4_val_2_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02=
;bootstrapped_ddqn_head_3_of_4/val_2_3/MatMul/ReadVariableOp
,bootstrapped_ddqn_head_3_of_4/val_2_3/MatMulMatMul8bootstrapped_ddqn_head_3_of_4/val_1_3/Relu:activations:0Cbootstrapped_ddqn_head_3_of_4/val_2_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,bootstrapped_ddqn_head_3_of_4/val_2_3/MatMulþ
<bootstrapped_ddqn_head_3_of_4/val_2_3/BiasAdd/ReadVariableOpReadVariableOpEbootstrapped_ddqn_head_3_of_4_val_2_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02>
<bootstrapped_ddqn_head_3_of_4/val_2_3/BiasAdd/ReadVariableOp
-bootstrapped_ddqn_head_3_of_4/val_2_3/BiasAddBiasAdd6bootstrapped_ddqn_head_3_of_4/val_2_3/MatMul:product:0Dbootstrapped_ddqn_head_3_of_4/val_2_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-bootstrapped_ddqn_head_3_of_4/val_2_3/BiasAdd£
:bootstrapped_ddqn_head_3_of_4/tf.__operators__.add_3/AddV2AddV26bootstrapped_ddqn_head_3_of_4/val_2_3/BiasAdd:output:06bootstrapped_ddqn_head_3_of_4/adv_2_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:bootstrapped_ddqn_head_3_of_4/tf.__operators__.add_3/AddV2Ú
Jbootstrapped_ddqn_head_3_of_4/tf.math.reduce_mean_3/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2L
Jbootstrapped_ddqn_head_3_of_4/tf.math.reduce_mean_3/Mean/reduction_indicesÌ
8bootstrapped_ddqn_head_3_of_4/tf.math.reduce_mean_3/MeanMean6bootstrapped_ddqn_head_3_of_4/adv_2_3/BiasAdd:output:0Sbootstrapped_ddqn_head_3_of_4/tf.math.reduce_mean_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2:
8bootstrapped_ddqn_head_3_of_4/tf.math.reduce_mean_3/Mean¨
4bootstrapped_ddqn_head_3_of_4/tf.math.subtract_3/SubSub>bootstrapped_ddqn_head_3_of_4/tf.__operators__.add_3/AddV2:z:0Abootstrapped_ddqn_head_3_of_4/tf.math.reduce_mean_3/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4bootstrapped_ddqn_head_3_of_4/tf.math.subtract_3/Sub÷
IdentityIdentity8bootstrapped_ddqn_head_3_of_4/tf.math.subtract_3/Sub:z:0=^bootstrapped_ddqn_head_3_of_4/adv_0_3/BiasAdd/ReadVariableOp<^bootstrapped_ddqn_head_3_of_4/adv_0_3/MatMul/ReadVariableOp=^bootstrapped_ddqn_head_3_of_4/adv_1_3/BiasAdd/ReadVariableOp<^bootstrapped_ddqn_head_3_of_4/adv_1_3/MatMul/ReadVariableOp=^bootstrapped_ddqn_head_3_of_4/adv_2_3/BiasAdd/ReadVariableOp<^bootstrapped_ddqn_head_3_of_4/adv_2_3/MatMul/ReadVariableOp=^bootstrapped_ddqn_head_3_of_4/dense_0/BiasAdd/ReadVariableOp<^bootstrapped_ddqn_head_3_of_4/dense_0/MatMul/ReadVariableOp=^bootstrapped_ddqn_head_3_of_4/val_0_3/BiasAdd/ReadVariableOp<^bootstrapped_ddqn_head_3_of_4/val_0_3/MatMul/ReadVariableOp=^bootstrapped_ddqn_head_3_of_4/val_1_3/BiasAdd/ReadVariableOp<^bootstrapped_ddqn_head_3_of_4/val_1_3/MatMul/ReadVariableOp=^bootstrapped_ddqn_head_3_of_4/val_2_3/BiasAdd/ReadVariableOp<^bootstrapped_ddqn_head_3_of_4/val_2_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ?: : : : : : : : : : : : : : 2|
<bootstrapped_ddqn_head_3_of_4/adv_0_3/BiasAdd/ReadVariableOp<bootstrapped_ddqn_head_3_of_4/adv_0_3/BiasAdd/ReadVariableOp2z
;bootstrapped_ddqn_head_3_of_4/adv_0_3/MatMul/ReadVariableOp;bootstrapped_ddqn_head_3_of_4/adv_0_3/MatMul/ReadVariableOp2|
<bootstrapped_ddqn_head_3_of_4/adv_1_3/BiasAdd/ReadVariableOp<bootstrapped_ddqn_head_3_of_4/adv_1_3/BiasAdd/ReadVariableOp2z
;bootstrapped_ddqn_head_3_of_4/adv_1_3/MatMul/ReadVariableOp;bootstrapped_ddqn_head_3_of_4/adv_1_3/MatMul/ReadVariableOp2|
<bootstrapped_ddqn_head_3_of_4/adv_2_3/BiasAdd/ReadVariableOp<bootstrapped_ddqn_head_3_of_4/adv_2_3/BiasAdd/ReadVariableOp2z
;bootstrapped_ddqn_head_3_of_4/adv_2_3/MatMul/ReadVariableOp;bootstrapped_ddqn_head_3_of_4/adv_2_3/MatMul/ReadVariableOp2|
<bootstrapped_ddqn_head_3_of_4/dense_0/BiasAdd/ReadVariableOp<bootstrapped_ddqn_head_3_of_4/dense_0/BiasAdd/ReadVariableOp2z
;bootstrapped_ddqn_head_3_of_4/dense_0/MatMul/ReadVariableOp;bootstrapped_ddqn_head_3_of_4/dense_0/MatMul/ReadVariableOp2|
<bootstrapped_ddqn_head_3_of_4/val_0_3/BiasAdd/ReadVariableOp<bootstrapped_ddqn_head_3_of_4/val_0_3/BiasAdd/ReadVariableOp2z
;bootstrapped_ddqn_head_3_of_4/val_0_3/MatMul/ReadVariableOp;bootstrapped_ddqn_head_3_of_4/val_0_3/MatMul/ReadVariableOp2|
<bootstrapped_ddqn_head_3_of_4/val_1_3/BiasAdd/ReadVariableOp<bootstrapped_ddqn_head_3_of_4/val_1_3/BiasAdd/ReadVariableOp2z
;bootstrapped_ddqn_head_3_of_4/val_1_3/MatMul/ReadVariableOp;bootstrapped_ddqn_head_3_of_4/val_1_3/MatMul/ReadVariableOp2|
<bootstrapped_ddqn_head_3_of_4/val_2_3/BiasAdd/ReadVariableOp<bootstrapped_ddqn_head_3_of_4/val_2_3/BiasAdd/ReadVariableOp2z
;bootstrapped_ddqn_head_3_of_4/val_2_3/MatMul/ReadVariableOp;bootstrapped_ddqn_head_3_of_4/val_2_3/MatMul/ReadVariableOp:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?
%
_user_specified_nameobservation
²

ø
F__inference_adv_0_3_layer_call_and_return_conditional_losses_263224782

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
F
Â

A__inference_bootstrapped_ddqn_head_3_of_4_layer_call_fn_263224727

inputs9
&dense_0_matmul_readvariableop_resource:	?6
'dense_0_biasadd_readvariableop_resource:	9
&adv_0_3_matmul_readvariableop_resource:	@5
'adv_0_3_biasadd_readvariableop_resource:@9
&val_0_3_matmul_readvariableop_resource:	@5
'val_0_3_biasadd_readvariableop_resource:@8
&adv_1_3_matmul_readvariableop_resource:@ 5
'adv_1_3_biasadd_readvariableop_resource: 8
&val_1_3_matmul_readvariableop_resource:@ 5
'val_1_3_biasadd_readvariableop_resource: 8
&adv_2_3_matmul_readvariableop_resource: 5
'adv_2_3_biasadd_readvariableop_resource:8
&val_2_3_matmul_readvariableop_resource: 5
'val_2_3_biasadd_readvariableop_resource:
identity¢adv_0_3/BiasAdd/ReadVariableOp¢adv_0_3/MatMul/ReadVariableOp¢adv_1_3/BiasAdd/ReadVariableOp¢adv_1_3/MatMul/ReadVariableOp¢adv_2_3/BiasAdd/ReadVariableOp¢adv_2_3/MatMul/ReadVariableOp¢dense_0/BiasAdd/ReadVariableOp¢dense_0/MatMul/ReadVariableOp¢val_0_3/BiasAdd/ReadVariableOp¢val_0_3/MatMul/ReadVariableOp¢val_1_3/BiasAdd/ReadVariableOp¢val_1_3/MatMul/ReadVariableOp¢val_2_3/BiasAdd/ReadVariableOp¢val_2_3/MatMul/ReadVariableOp¦
dense_0/MatMul/ReadVariableOpReadVariableOp&dense_0_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_0/MatMul/ReadVariableOp
dense_0/MatMulMatMulinputs%dense_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_0/MatMul¥
dense_0/BiasAdd/ReadVariableOpReadVariableOp'dense_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_0/BiasAdd/ReadVariableOp¢
dense_0/BiasAddBiasAdddense_0/MatMul:product:0&dense_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_0/BiasAddq
dense_0/ReluReludense_0/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_0/Relu¦
adv_0_3/MatMul/ReadVariableOpReadVariableOp&adv_0_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
adv_0_3/MatMul/ReadVariableOp
adv_0_3/MatMulMatMuldense_0/Relu:activations:0%adv_0_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_0_3/MatMul¤
adv_0_3/BiasAdd/ReadVariableOpReadVariableOp'adv_0_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
adv_0_3/BiasAdd/ReadVariableOp¡
adv_0_3/BiasAddBiasAddadv_0_3/MatMul:product:0&adv_0_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_0_3/BiasAddp
adv_0_3/ReluReluadv_0_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_0_3/Relu¦
val_0_3/MatMul/ReadVariableOpReadVariableOp&val_0_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
val_0_3/MatMul/ReadVariableOp
val_0_3/MatMulMatMuldense_0/Relu:activations:0%val_0_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_0_3/MatMul¤
val_0_3/BiasAdd/ReadVariableOpReadVariableOp'val_0_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
val_0_3/BiasAdd/ReadVariableOp¡
val_0_3/BiasAddBiasAddval_0_3/MatMul:product:0&val_0_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_0_3/BiasAddp
val_0_3/ReluReluval_0_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_0_3/Relu¥
adv_1_3/MatMul/ReadVariableOpReadVariableOp&adv_1_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
adv_1_3/MatMul/ReadVariableOp
adv_1_3/MatMulMatMuladv_0_3/Relu:activations:0%adv_1_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
adv_1_3/MatMul¤
adv_1_3/BiasAdd/ReadVariableOpReadVariableOp'adv_1_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
adv_1_3/BiasAdd/ReadVariableOp¡
adv_1_3/BiasAddBiasAddadv_1_3/MatMul:product:0&adv_1_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
adv_1_3/BiasAddp
adv_1_3/ReluReluadv_1_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
adv_1_3/Relu¥
val_1_3/MatMul/ReadVariableOpReadVariableOp&val_1_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
val_1_3/MatMul/ReadVariableOp
val_1_3/MatMulMatMulval_0_3/Relu:activations:0%val_1_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
val_1_3/MatMul¤
val_1_3/BiasAdd/ReadVariableOpReadVariableOp'val_1_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
val_1_3/BiasAdd/ReadVariableOp¡
val_1_3/BiasAddBiasAddval_1_3/MatMul:product:0&val_1_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
val_1_3/BiasAddp
val_1_3/ReluReluval_1_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
val_1_3/Relu¥
adv_2_3/MatMul/ReadVariableOpReadVariableOp&adv_2_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
adv_2_3/MatMul/ReadVariableOp
adv_2_3/MatMulMatMuladv_1_3/Relu:activations:0%adv_2_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adv_2_3/MatMul¤
adv_2_3/BiasAdd/ReadVariableOpReadVariableOp'adv_2_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
adv_2_3/BiasAdd/ReadVariableOp¡
adv_2_3/BiasAddBiasAddadv_2_3/MatMul:product:0&adv_2_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adv_2_3/BiasAdd¥
val_2_3/MatMul/ReadVariableOpReadVariableOp&val_2_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
val_2_3/MatMul/ReadVariableOp
val_2_3/MatMulMatMulval_1_3/Relu:activations:0%val_2_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
val_2_3/MatMul¤
val_2_3/BiasAdd/ReadVariableOpReadVariableOp'val_2_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
val_2_3/BiasAdd/ReadVariableOp¡
val_2_3/BiasAddBiasAddval_2_3/MatMul:product:0&val_2_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
val_2_3/BiasAdd«
tf.__operators__.add_3/AddV2AddV2val_2_3/BiasAdd:output:0adv_2_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_3/AddV2
,tf.math.reduce_mean_3/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,tf.math.reduce_mean_3/Mean/reduction_indicesÔ
tf.math.reduce_mean_3/MeanMeanadv_2_3/BiasAdd:output:05tf.math.reduce_mean_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
tf.math.reduce_mean_3/Mean°
tf.math.subtract_3/SubSub tf.__operators__.add_3/AddV2:z:0#tf.math.reduce_mean_3/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.subtract_3/Subµ
IdentityIdentitytf.math.subtract_3/Sub:z:0^adv_0_3/BiasAdd/ReadVariableOp^adv_0_3/MatMul/ReadVariableOp^adv_1_3/BiasAdd/ReadVariableOp^adv_1_3/MatMul/ReadVariableOp^adv_2_3/BiasAdd/ReadVariableOp^adv_2_3/MatMul/ReadVariableOp^dense_0/BiasAdd/ReadVariableOp^dense_0/MatMul/ReadVariableOp^val_0_3/BiasAdd/ReadVariableOp^val_0_3/MatMul/ReadVariableOp^val_1_3/BiasAdd/ReadVariableOp^val_1_3/MatMul/ReadVariableOp^val_2_3/BiasAdd/ReadVariableOp^val_2_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ?: : : : : : : : : : : : : : 2@
adv_0_3/BiasAdd/ReadVariableOpadv_0_3/BiasAdd/ReadVariableOp2>
adv_0_3/MatMul/ReadVariableOpadv_0_3/MatMul/ReadVariableOp2@
adv_1_3/BiasAdd/ReadVariableOpadv_1_3/BiasAdd/ReadVariableOp2>
adv_1_3/MatMul/ReadVariableOpadv_1_3/MatMul/ReadVariableOp2@
adv_2_3/BiasAdd/ReadVariableOpadv_2_3/BiasAdd/ReadVariableOp2>
adv_2_3/MatMul/ReadVariableOpadv_2_3/MatMul/ReadVariableOp2@
dense_0/BiasAdd/ReadVariableOpdense_0/BiasAdd/ReadVariableOp2>
dense_0/MatMul/ReadVariableOpdense_0/MatMul/ReadVariableOp2@
val_0_3/BiasAdd/ReadVariableOpval_0_3/BiasAdd/ReadVariableOp2>
val_0_3/MatMul/ReadVariableOpval_0_3/MatMul/ReadVariableOp2@
val_1_3/BiasAdd/ReadVariableOpval_1_3/BiasAdd/ReadVariableOp2>
val_1_3/MatMul/ReadVariableOpval_1_3/MatMul/ReadVariableOp2@
val_2_3/BiasAdd/ReadVariableOpval_2_3/BiasAdd/ReadVariableOp2>
val_2_3/MatMul/ReadVariableOpval_2_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?
 
_user_specified_nameinputs
¬F
â

\__inference_bootstrapped_ddqn_head_3_of_4_layer_call_and_return_conditional_losses_263224466
observation9
&dense_0_matmul_readvariableop_resource:	?6
'dense_0_biasadd_readvariableop_resource:	9
&adv_0_3_matmul_readvariableop_resource:	@5
'adv_0_3_biasadd_readvariableop_resource:@9
&val_0_3_matmul_readvariableop_resource:	@5
'val_0_3_biasadd_readvariableop_resource:@8
&adv_1_3_matmul_readvariableop_resource:@ 5
'adv_1_3_biasadd_readvariableop_resource: 8
&val_1_3_matmul_readvariableop_resource:@ 5
'val_1_3_biasadd_readvariableop_resource: 8
&adv_2_3_matmul_readvariableop_resource: 5
'adv_2_3_biasadd_readvariableop_resource:8
&val_2_3_matmul_readvariableop_resource: 5
'val_2_3_biasadd_readvariableop_resource:
identity¢adv_0_3/BiasAdd/ReadVariableOp¢adv_0_3/MatMul/ReadVariableOp¢adv_1_3/BiasAdd/ReadVariableOp¢adv_1_3/MatMul/ReadVariableOp¢adv_2_3/BiasAdd/ReadVariableOp¢adv_2_3/MatMul/ReadVariableOp¢dense_0/BiasAdd/ReadVariableOp¢dense_0/MatMul/ReadVariableOp¢val_0_3/BiasAdd/ReadVariableOp¢val_0_3/MatMul/ReadVariableOp¢val_1_3/BiasAdd/ReadVariableOp¢val_1_3/MatMul/ReadVariableOp¢val_2_3/BiasAdd/ReadVariableOp¢val_2_3/MatMul/ReadVariableOp¦
dense_0/MatMul/ReadVariableOpReadVariableOp&dense_0_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_0/MatMul/ReadVariableOp
dense_0/MatMulMatMulobservation%dense_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_0/MatMul¥
dense_0/BiasAdd/ReadVariableOpReadVariableOp'dense_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_0/BiasAdd/ReadVariableOp¢
dense_0/BiasAddBiasAdddense_0/MatMul:product:0&dense_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_0/BiasAddq
dense_0/ReluReludense_0/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_0/Relu¦
adv_0_3/MatMul/ReadVariableOpReadVariableOp&adv_0_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
adv_0_3/MatMul/ReadVariableOp
adv_0_3/MatMulMatMuldense_0/Relu:activations:0%adv_0_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_0_3/MatMul¤
adv_0_3/BiasAdd/ReadVariableOpReadVariableOp'adv_0_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
adv_0_3/BiasAdd/ReadVariableOp¡
adv_0_3/BiasAddBiasAddadv_0_3/MatMul:product:0&adv_0_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_0_3/BiasAddp
adv_0_3/ReluReluadv_0_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_0_3/Relu¦
val_0_3/MatMul/ReadVariableOpReadVariableOp&val_0_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
val_0_3/MatMul/ReadVariableOp
val_0_3/MatMulMatMuldense_0/Relu:activations:0%val_0_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_0_3/MatMul¤
val_0_3/BiasAdd/ReadVariableOpReadVariableOp'val_0_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
val_0_3/BiasAdd/ReadVariableOp¡
val_0_3/BiasAddBiasAddval_0_3/MatMul:product:0&val_0_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_0_3/BiasAddp
val_0_3/ReluReluval_0_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_0_3/Relu¥
adv_1_3/MatMul/ReadVariableOpReadVariableOp&adv_1_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
adv_1_3/MatMul/ReadVariableOp
adv_1_3/MatMulMatMuladv_0_3/Relu:activations:0%adv_1_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
adv_1_3/MatMul¤
adv_1_3/BiasAdd/ReadVariableOpReadVariableOp'adv_1_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
adv_1_3/BiasAdd/ReadVariableOp¡
adv_1_3/BiasAddBiasAddadv_1_3/MatMul:product:0&adv_1_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
adv_1_3/BiasAddp
adv_1_3/ReluReluadv_1_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
adv_1_3/Relu¥
val_1_3/MatMul/ReadVariableOpReadVariableOp&val_1_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
val_1_3/MatMul/ReadVariableOp
val_1_3/MatMulMatMulval_0_3/Relu:activations:0%val_1_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
val_1_3/MatMul¤
val_1_3/BiasAdd/ReadVariableOpReadVariableOp'val_1_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
val_1_3/BiasAdd/ReadVariableOp¡
val_1_3/BiasAddBiasAddval_1_3/MatMul:product:0&val_1_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
val_1_3/BiasAddp
val_1_3/ReluReluval_1_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
val_1_3/Relu¥
adv_2_3/MatMul/ReadVariableOpReadVariableOp&adv_2_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
adv_2_3/MatMul/ReadVariableOp
adv_2_3/MatMulMatMuladv_1_3/Relu:activations:0%adv_2_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adv_2_3/MatMul¤
adv_2_3/BiasAdd/ReadVariableOpReadVariableOp'adv_2_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
adv_2_3/BiasAdd/ReadVariableOp¡
adv_2_3/BiasAddBiasAddadv_2_3/MatMul:product:0&adv_2_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adv_2_3/BiasAdd¥
val_2_3/MatMul/ReadVariableOpReadVariableOp&val_2_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
val_2_3/MatMul/ReadVariableOp
val_2_3/MatMulMatMulval_1_3/Relu:activations:0%val_2_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
val_2_3/MatMul¤
val_2_3/BiasAdd/ReadVariableOpReadVariableOp'val_2_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
val_2_3/BiasAdd/ReadVariableOp¡
val_2_3/BiasAddBiasAddval_2_3/MatMul:product:0&val_2_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
val_2_3/BiasAdd«
tf.__operators__.add_3/AddV2AddV2val_2_3/BiasAdd:output:0adv_2_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_3/AddV2
,tf.math.reduce_mean_3/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,tf.math.reduce_mean_3/Mean/reduction_indicesÔ
tf.math.reduce_mean_3/MeanMeanadv_2_3/BiasAdd:output:05tf.math.reduce_mean_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
tf.math.reduce_mean_3/Mean°
tf.math.subtract_3/SubSub tf.__operators__.add_3/AddV2:z:0#tf.math.reduce_mean_3/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.subtract_3/Subµ
IdentityIdentitytf.math.subtract_3/Sub:z:0^adv_0_3/BiasAdd/ReadVariableOp^adv_0_3/MatMul/ReadVariableOp^adv_1_3/BiasAdd/ReadVariableOp^adv_1_3/MatMul/ReadVariableOp^adv_2_3/BiasAdd/ReadVariableOp^adv_2_3/MatMul/ReadVariableOp^dense_0/BiasAdd/ReadVariableOp^dense_0/MatMul/ReadVariableOp^val_0_3/BiasAdd/ReadVariableOp^val_0_3/MatMul/ReadVariableOp^val_1_3/BiasAdd/ReadVariableOp^val_1_3/MatMul/ReadVariableOp^val_2_3/BiasAdd/ReadVariableOp^val_2_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ?: : : : : : : : : : : : : : 2@
adv_0_3/BiasAdd/ReadVariableOpadv_0_3/BiasAdd/ReadVariableOp2>
adv_0_3/MatMul/ReadVariableOpadv_0_3/MatMul/ReadVariableOp2@
adv_1_3/BiasAdd/ReadVariableOpadv_1_3/BiasAdd/ReadVariableOp2>
adv_1_3/MatMul/ReadVariableOpadv_1_3/MatMul/ReadVariableOp2@
adv_2_3/BiasAdd/ReadVariableOpadv_2_3/BiasAdd/ReadVariableOp2>
adv_2_3/MatMul/ReadVariableOpadv_2_3/MatMul/ReadVariableOp2@
dense_0/BiasAdd/ReadVariableOpdense_0/BiasAdd/ReadVariableOp2>
dense_0/MatMul/ReadVariableOpdense_0/MatMul/ReadVariableOp2@
val_0_3/BiasAdd/ReadVariableOpval_0_3/BiasAdd/ReadVariableOp2>
val_0_3/MatMul/ReadVariableOpval_0_3/MatMul/ReadVariableOp2@
val_1_3/BiasAdd/ReadVariableOpval_1_3/BiasAdd/ReadVariableOp2>
val_1_3/MatMul/ReadVariableOpval_1_3/MatMul/ReadVariableOp2@
val_2_3/BiasAdd/ReadVariableOpval_2_3/BiasAdd/ReadVariableOp2>
val_2_3/MatMul/ReadVariableOpval_2_3/MatMul/ReadVariableOp:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?
%
_user_specified_nameobservation
²

ø
F__inference_val_0_3_layer_call_and_return_conditional_losses_263224760

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®

÷
F__inference_val_1_3_layer_call_and_return_conditional_losses_263224804

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


Ý
+__inference_adv_0_3_layer_call_fn_263224793

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
F
Â

A__inference_bootstrapped_ddqn_head_3_of_4_layer_call_fn_263224672

inputs9
&dense_0_matmul_readvariableop_resource:	?6
'dense_0_biasadd_readvariableop_resource:	9
&adv_0_3_matmul_readvariableop_resource:	@5
'adv_0_3_biasadd_readvariableop_resource:@9
&val_0_3_matmul_readvariableop_resource:	@5
'val_0_3_biasadd_readvariableop_resource:@8
&adv_1_3_matmul_readvariableop_resource:@ 5
'adv_1_3_biasadd_readvariableop_resource: 8
&val_1_3_matmul_readvariableop_resource:@ 5
'val_1_3_biasadd_readvariableop_resource: 8
&adv_2_3_matmul_readvariableop_resource: 5
'adv_2_3_biasadd_readvariableop_resource:8
&val_2_3_matmul_readvariableop_resource: 5
'val_2_3_biasadd_readvariableop_resource:
identity¢adv_0_3/BiasAdd/ReadVariableOp¢adv_0_3/MatMul/ReadVariableOp¢adv_1_3/BiasAdd/ReadVariableOp¢adv_1_3/MatMul/ReadVariableOp¢adv_2_3/BiasAdd/ReadVariableOp¢adv_2_3/MatMul/ReadVariableOp¢dense_0/BiasAdd/ReadVariableOp¢dense_0/MatMul/ReadVariableOp¢val_0_3/BiasAdd/ReadVariableOp¢val_0_3/MatMul/ReadVariableOp¢val_1_3/BiasAdd/ReadVariableOp¢val_1_3/MatMul/ReadVariableOp¢val_2_3/BiasAdd/ReadVariableOp¢val_2_3/MatMul/ReadVariableOp¦
dense_0/MatMul/ReadVariableOpReadVariableOp&dense_0_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_0/MatMul/ReadVariableOp
dense_0/MatMulMatMulinputs%dense_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_0/MatMul¥
dense_0/BiasAdd/ReadVariableOpReadVariableOp'dense_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_0/BiasAdd/ReadVariableOp¢
dense_0/BiasAddBiasAdddense_0/MatMul:product:0&dense_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_0/BiasAddq
dense_0/ReluReludense_0/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_0/Relu¦
adv_0_3/MatMul/ReadVariableOpReadVariableOp&adv_0_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
adv_0_3/MatMul/ReadVariableOp
adv_0_3/MatMulMatMuldense_0/Relu:activations:0%adv_0_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_0_3/MatMul¤
adv_0_3/BiasAdd/ReadVariableOpReadVariableOp'adv_0_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
adv_0_3/BiasAdd/ReadVariableOp¡
adv_0_3/BiasAddBiasAddadv_0_3/MatMul:product:0&adv_0_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_0_3/BiasAddp
adv_0_3/ReluReluadv_0_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_0_3/Relu¦
val_0_3/MatMul/ReadVariableOpReadVariableOp&val_0_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
val_0_3/MatMul/ReadVariableOp
val_0_3/MatMulMatMuldense_0/Relu:activations:0%val_0_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_0_3/MatMul¤
val_0_3/BiasAdd/ReadVariableOpReadVariableOp'val_0_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
val_0_3/BiasAdd/ReadVariableOp¡
val_0_3/BiasAddBiasAddval_0_3/MatMul:product:0&val_0_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_0_3/BiasAddp
val_0_3/ReluReluval_0_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_0_3/Relu¥
adv_1_3/MatMul/ReadVariableOpReadVariableOp&adv_1_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
adv_1_3/MatMul/ReadVariableOp
adv_1_3/MatMulMatMuladv_0_3/Relu:activations:0%adv_1_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
adv_1_3/MatMul¤
adv_1_3/BiasAdd/ReadVariableOpReadVariableOp'adv_1_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
adv_1_3/BiasAdd/ReadVariableOp¡
adv_1_3/BiasAddBiasAddadv_1_3/MatMul:product:0&adv_1_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
adv_1_3/BiasAddp
adv_1_3/ReluReluadv_1_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
adv_1_3/Relu¥
val_1_3/MatMul/ReadVariableOpReadVariableOp&val_1_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
val_1_3/MatMul/ReadVariableOp
val_1_3/MatMulMatMulval_0_3/Relu:activations:0%val_1_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
val_1_3/MatMul¤
val_1_3/BiasAdd/ReadVariableOpReadVariableOp'val_1_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
val_1_3/BiasAdd/ReadVariableOp¡
val_1_3/BiasAddBiasAddval_1_3/MatMul:product:0&val_1_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
val_1_3/BiasAddp
val_1_3/ReluReluval_1_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
val_1_3/Relu¥
adv_2_3/MatMul/ReadVariableOpReadVariableOp&adv_2_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
adv_2_3/MatMul/ReadVariableOp
adv_2_3/MatMulMatMuladv_1_3/Relu:activations:0%adv_2_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adv_2_3/MatMul¤
adv_2_3/BiasAdd/ReadVariableOpReadVariableOp'adv_2_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
adv_2_3/BiasAdd/ReadVariableOp¡
adv_2_3/BiasAddBiasAddadv_2_3/MatMul:product:0&adv_2_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adv_2_3/BiasAdd¥
val_2_3/MatMul/ReadVariableOpReadVariableOp&val_2_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
val_2_3/MatMul/ReadVariableOp
val_2_3/MatMulMatMulval_1_3/Relu:activations:0%val_2_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
val_2_3/MatMul¤
val_2_3/BiasAdd/ReadVariableOpReadVariableOp'val_2_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
val_2_3/BiasAdd/ReadVariableOp¡
val_2_3/BiasAddBiasAddval_2_3/MatMul:product:0&val_2_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
val_2_3/BiasAdd«
tf.__operators__.add_3/AddV2AddV2val_2_3/BiasAdd:output:0adv_2_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_3/AddV2
,tf.math.reduce_mean_3/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,tf.math.reduce_mean_3/Mean/reduction_indicesÔ
tf.math.reduce_mean_3/MeanMeanadv_2_3/BiasAdd:output:05tf.math.reduce_mean_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
tf.math.reduce_mean_3/Mean°
tf.math.subtract_3/SubSub tf.__operators__.add_3/AddV2:z:0#tf.math.reduce_mean_3/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.subtract_3/Subµ
IdentityIdentitytf.math.subtract_3/Sub:z:0^adv_0_3/BiasAdd/ReadVariableOp^adv_0_3/MatMul/ReadVariableOp^adv_1_3/BiasAdd/ReadVariableOp^adv_1_3/MatMul/ReadVariableOp^adv_2_3/BiasAdd/ReadVariableOp^adv_2_3/MatMul/ReadVariableOp^dense_0/BiasAdd/ReadVariableOp^dense_0/MatMul/ReadVariableOp^val_0_3/BiasAdd/ReadVariableOp^val_0_3/MatMul/ReadVariableOp^val_1_3/BiasAdd/ReadVariableOp^val_1_3/MatMul/ReadVariableOp^val_2_3/BiasAdd/ReadVariableOp^val_2_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ?: : : : : : : : : : : : : : 2@
adv_0_3/BiasAdd/ReadVariableOpadv_0_3/BiasAdd/ReadVariableOp2>
adv_0_3/MatMul/ReadVariableOpadv_0_3/MatMul/ReadVariableOp2@
adv_1_3/BiasAdd/ReadVariableOpadv_1_3/BiasAdd/ReadVariableOp2>
adv_1_3/MatMul/ReadVariableOpadv_1_3/MatMul/ReadVariableOp2@
adv_2_3/BiasAdd/ReadVariableOpadv_2_3/BiasAdd/ReadVariableOp2>
adv_2_3/MatMul/ReadVariableOpadv_2_3/MatMul/ReadVariableOp2@
dense_0/BiasAdd/ReadVariableOpdense_0/BiasAdd/ReadVariableOp2>
dense_0/MatMul/ReadVariableOpdense_0/MatMul/ReadVariableOp2@
val_0_3/BiasAdd/ReadVariableOpval_0_3/BiasAdd/ReadVariableOp2>
val_0_3/MatMul/ReadVariableOpval_0_3/MatMul/ReadVariableOp2@
val_1_3/BiasAdd/ReadVariableOpval_1_3/BiasAdd/ReadVariableOp2>
val_1_3/MatMul/ReadVariableOpval_1_3/MatMul/ReadVariableOp2@
val_2_3/BiasAdd/ReadVariableOpval_2_3/BiasAdd/ReadVariableOp2>
val_2_3/MatMul/ReadVariableOpval_2_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?
 
_user_specified_nameinputs


Ü
+__inference_adv_1_3_layer_call_fn_263224837

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ò	
÷
F__inference_adv_2_3_layer_call_and_return_conditional_losses_263224867

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
F
Ç

A__inference_bootstrapped_ddqn_head_3_of_4_layer_call_fn_263223916
observation9
&dense_0_matmul_readvariableop_resource:	?6
'dense_0_biasadd_readvariableop_resource:	9
&adv_0_3_matmul_readvariableop_resource:	@5
'adv_0_3_biasadd_readvariableop_resource:@9
&val_0_3_matmul_readvariableop_resource:	@5
'val_0_3_biasadd_readvariableop_resource:@8
&adv_1_3_matmul_readvariableop_resource:@ 5
'adv_1_3_biasadd_readvariableop_resource: 8
&val_1_3_matmul_readvariableop_resource:@ 5
'val_1_3_biasadd_readvariableop_resource: 8
&adv_2_3_matmul_readvariableop_resource: 5
'adv_2_3_biasadd_readvariableop_resource:8
&val_2_3_matmul_readvariableop_resource: 5
'val_2_3_biasadd_readvariableop_resource:
identity¢adv_0_3/BiasAdd/ReadVariableOp¢adv_0_3/MatMul/ReadVariableOp¢adv_1_3/BiasAdd/ReadVariableOp¢adv_1_3/MatMul/ReadVariableOp¢adv_2_3/BiasAdd/ReadVariableOp¢adv_2_3/MatMul/ReadVariableOp¢dense_0/BiasAdd/ReadVariableOp¢dense_0/MatMul/ReadVariableOp¢val_0_3/BiasAdd/ReadVariableOp¢val_0_3/MatMul/ReadVariableOp¢val_1_3/BiasAdd/ReadVariableOp¢val_1_3/MatMul/ReadVariableOp¢val_2_3/BiasAdd/ReadVariableOp¢val_2_3/MatMul/ReadVariableOp¦
dense_0/MatMul/ReadVariableOpReadVariableOp&dense_0_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_0/MatMul/ReadVariableOp
dense_0/MatMulMatMulobservation%dense_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_0/MatMul¥
dense_0/BiasAdd/ReadVariableOpReadVariableOp'dense_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_0/BiasAdd/ReadVariableOp¢
dense_0/BiasAddBiasAdddense_0/MatMul:product:0&dense_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_0/BiasAddq
dense_0/ReluReludense_0/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_0/Relu¦
adv_0_3/MatMul/ReadVariableOpReadVariableOp&adv_0_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
adv_0_3/MatMul/ReadVariableOp
adv_0_3/MatMulMatMuldense_0/Relu:activations:0%adv_0_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_0_3/MatMul¤
adv_0_3/BiasAdd/ReadVariableOpReadVariableOp'adv_0_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
adv_0_3/BiasAdd/ReadVariableOp¡
adv_0_3/BiasAddBiasAddadv_0_3/MatMul:product:0&adv_0_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_0_3/BiasAddp
adv_0_3/ReluReluadv_0_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_0_3/Relu¦
val_0_3/MatMul/ReadVariableOpReadVariableOp&val_0_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
val_0_3/MatMul/ReadVariableOp
val_0_3/MatMulMatMuldense_0/Relu:activations:0%val_0_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_0_3/MatMul¤
val_0_3/BiasAdd/ReadVariableOpReadVariableOp'val_0_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
val_0_3/BiasAdd/ReadVariableOp¡
val_0_3/BiasAddBiasAddval_0_3/MatMul:product:0&val_0_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_0_3/BiasAddp
val_0_3/ReluReluval_0_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_0_3/Relu¥
adv_1_3/MatMul/ReadVariableOpReadVariableOp&adv_1_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
adv_1_3/MatMul/ReadVariableOp
adv_1_3/MatMulMatMuladv_0_3/Relu:activations:0%adv_1_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
adv_1_3/MatMul¤
adv_1_3/BiasAdd/ReadVariableOpReadVariableOp'adv_1_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
adv_1_3/BiasAdd/ReadVariableOp¡
adv_1_3/BiasAddBiasAddadv_1_3/MatMul:product:0&adv_1_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
adv_1_3/BiasAddp
adv_1_3/ReluReluadv_1_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
adv_1_3/Relu¥
val_1_3/MatMul/ReadVariableOpReadVariableOp&val_1_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
val_1_3/MatMul/ReadVariableOp
val_1_3/MatMulMatMulval_0_3/Relu:activations:0%val_1_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
val_1_3/MatMul¤
val_1_3/BiasAdd/ReadVariableOpReadVariableOp'val_1_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
val_1_3/BiasAdd/ReadVariableOp¡
val_1_3/BiasAddBiasAddval_1_3/MatMul:product:0&val_1_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
val_1_3/BiasAddp
val_1_3/ReluReluval_1_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
val_1_3/Relu¥
adv_2_3/MatMul/ReadVariableOpReadVariableOp&adv_2_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
adv_2_3/MatMul/ReadVariableOp
adv_2_3/MatMulMatMuladv_1_3/Relu:activations:0%adv_2_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adv_2_3/MatMul¤
adv_2_3/BiasAdd/ReadVariableOpReadVariableOp'adv_2_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
adv_2_3/BiasAdd/ReadVariableOp¡
adv_2_3/BiasAddBiasAddadv_2_3/MatMul:product:0&adv_2_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adv_2_3/BiasAdd¥
val_2_3/MatMul/ReadVariableOpReadVariableOp&val_2_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
val_2_3/MatMul/ReadVariableOp
val_2_3/MatMulMatMulval_1_3/Relu:activations:0%val_2_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
val_2_3/MatMul¤
val_2_3/BiasAdd/ReadVariableOpReadVariableOp'val_2_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
val_2_3/BiasAdd/ReadVariableOp¡
val_2_3/BiasAddBiasAddval_2_3/MatMul:product:0&val_2_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
val_2_3/BiasAdd«
tf.__operators__.add_3/AddV2AddV2val_2_3/BiasAdd:output:0adv_2_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_3/AddV2
,tf.math.reduce_mean_3/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,tf.math.reduce_mean_3/Mean/reduction_indicesÔ
tf.math.reduce_mean_3/MeanMeanadv_2_3/BiasAdd:output:05tf.math.reduce_mean_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
tf.math.reduce_mean_3/Mean°
tf.math.subtract_3/SubSub tf.__operators__.add_3/AddV2:z:0#tf.math.reduce_mean_3/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.subtract_3/Subµ
IdentityIdentitytf.math.subtract_3/Sub:z:0^adv_0_3/BiasAdd/ReadVariableOp^adv_0_3/MatMul/ReadVariableOp^adv_1_3/BiasAdd/ReadVariableOp^adv_1_3/MatMul/ReadVariableOp^adv_2_3/BiasAdd/ReadVariableOp^adv_2_3/MatMul/ReadVariableOp^dense_0/BiasAdd/ReadVariableOp^dense_0/MatMul/ReadVariableOp^val_0_3/BiasAdd/ReadVariableOp^val_0_3/MatMul/ReadVariableOp^val_1_3/BiasAdd/ReadVariableOp^val_1_3/MatMul/ReadVariableOp^val_2_3/BiasAdd/ReadVariableOp^val_2_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ?: : : : : : : : : : : : : : 2@
adv_0_3/BiasAdd/ReadVariableOpadv_0_3/BiasAdd/ReadVariableOp2>
adv_0_3/MatMul/ReadVariableOpadv_0_3/MatMul/ReadVariableOp2@
adv_1_3/BiasAdd/ReadVariableOpadv_1_3/BiasAdd/ReadVariableOp2>
adv_1_3/MatMul/ReadVariableOpadv_1_3/MatMul/ReadVariableOp2@
adv_2_3/BiasAdd/ReadVariableOpadv_2_3/BiasAdd/ReadVariableOp2>
adv_2_3/MatMul/ReadVariableOpadv_2_3/MatMul/ReadVariableOp2@
dense_0/BiasAdd/ReadVariableOpdense_0/BiasAdd/ReadVariableOp2>
dense_0/MatMul/ReadVariableOpdense_0/MatMul/ReadVariableOp2@
val_0_3/BiasAdd/ReadVariableOpval_0_3/BiasAdd/ReadVariableOp2>
val_0_3/MatMul/ReadVariableOpval_0_3/MatMul/ReadVariableOp2@
val_1_3/BiasAdd/ReadVariableOpval_1_3/BiasAdd/ReadVariableOp2>
val_1_3/MatMul/ReadVariableOpval_1_3/MatMul/ReadVariableOp2@
val_2_3/BiasAdd/ReadVariableOpval_2_3/BiasAdd/ReadVariableOp2>
val_2_3/MatMul/ReadVariableOpval_2_3/MatMul/ReadVariableOp:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?
%
_user_specified_nameobservation


Þ
+__inference_dense_0_layer_call_fn_263224749

inputs1
matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?
 
_user_specified_nameinputs
¶

ù
F__inference_dense_0_layer_call_and_return_conditional_losses_263224738

inputs1
matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?
 
_user_specified_nameinputs
F
Ý

\__inference_bootstrapped_ddqn_head_3_of_4_layer_call_and_return_conditional_losses_263224562

inputs9
&dense_0_matmul_readvariableop_resource:	?6
'dense_0_biasadd_readvariableop_resource:	9
&adv_0_3_matmul_readvariableop_resource:	@5
'adv_0_3_biasadd_readvariableop_resource:@9
&val_0_3_matmul_readvariableop_resource:	@5
'val_0_3_biasadd_readvariableop_resource:@8
&adv_1_3_matmul_readvariableop_resource:@ 5
'adv_1_3_biasadd_readvariableop_resource: 8
&val_1_3_matmul_readvariableop_resource:@ 5
'val_1_3_biasadd_readvariableop_resource: 8
&adv_2_3_matmul_readvariableop_resource: 5
'adv_2_3_biasadd_readvariableop_resource:8
&val_2_3_matmul_readvariableop_resource: 5
'val_2_3_biasadd_readvariableop_resource:
identity¢adv_0_3/BiasAdd/ReadVariableOp¢adv_0_3/MatMul/ReadVariableOp¢adv_1_3/BiasAdd/ReadVariableOp¢adv_1_3/MatMul/ReadVariableOp¢adv_2_3/BiasAdd/ReadVariableOp¢adv_2_3/MatMul/ReadVariableOp¢dense_0/BiasAdd/ReadVariableOp¢dense_0/MatMul/ReadVariableOp¢val_0_3/BiasAdd/ReadVariableOp¢val_0_3/MatMul/ReadVariableOp¢val_1_3/BiasAdd/ReadVariableOp¢val_1_3/MatMul/ReadVariableOp¢val_2_3/BiasAdd/ReadVariableOp¢val_2_3/MatMul/ReadVariableOp¦
dense_0/MatMul/ReadVariableOpReadVariableOp&dense_0_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_0/MatMul/ReadVariableOp
dense_0/MatMulMatMulinputs%dense_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_0/MatMul¥
dense_0/BiasAdd/ReadVariableOpReadVariableOp'dense_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_0/BiasAdd/ReadVariableOp¢
dense_0/BiasAddBiasAdddense_0/MatMul:product:0&dense_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_0/BiasAddq
dense_0/ReluReludense_0/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_0/Relu¦
adv_0_3/MatMul/ReadVariableOpReadVariableOp&adv_0_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
adv_0_3/MatMul/ReadVariableOp
adv_0_3/MatMulMatMuldense_0/Relu:activations:0%adv_0_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_0_3/MatMul¤
adv_0_3/BiasAdd/ReadVariableOpReadVariableOp'adv_0_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
adv_0_3/BiasAdd/ReadVariableOp¡
adv_0_3/BiasAddBiasAddadv_0_3/MatMul:product:0&adv_0_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_0_3/BiasAddp
adv_0_3/ReluReluadv_0_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_0_3/Relu¦
val_0_3/MatMul/ReadVariableOpReadVariableOp&val_0_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
val_0_3/MatMul/ReadVariableOp
val_0_3/MatMulMatMuldense_0/Relu:activations:0%val_0_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_0_3/MatMul¤
val_0_3/BiasAdd/ReadVariableOpReadVariableOp'val_0_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
val_0_3/BiasAdd/ReadVariableOp¡
val_0_3/BiasAddBiasAddval_0_3/MatMul:product:0&val_0_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_0_3/BiasAddp
val_0_3/ReluReluval_0_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_0_3/Relu¥
adv_1_3/MatMul/ReadVariableOpReadVariableOp&adv_1_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
adv_1_3/MatMul/ReadVariableOp
adv_1_3/MatMulMatMuladv_0_3/Relu:activations:0%adv_1_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
adv_1_3/MatMul¤
adv_1_3/BiasAdd/ReadVariableOpReadVariableOp'adv_1_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
adv_1_3/BiasAdd/ReadVariableOp¡
adv_1_3/BiasAddBiasAddadv_1_3/MatMul:product:0&adv_1_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
adv_1_3/BiasAddp
adv_1_3/ReluReluadv_1_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
adv_1_3/Relu¥
val_1_3/MatMul/ReadVariableOpReadVariableOp&val_1_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
val_1_3/MatMul/ReadVariableOp
val_1_3/MatMulMatMulval_0_3/Relu:activations:0%val_1_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
val_1_3/MatMul¤
val_1_3/BiasAdd/ReadVariableOpReadVariableOp'val_1_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
val_1_3/BiasAdd/ReadVariableOp¡
val_1_3/BiasAddBiasAddval_1_3/MatMul:product:0&val_1_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
val_1_3/BiasAddp
val_1_3/ReluReluval_1_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
val_1_3/Relu¥
adv_2_3/MatMul/ReadVariableOpReadVariableOp&adv_2_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
adv_2_3/MatMul/ReadVariableOp
adv_2_3/MatMulMatMuladv_1_3/Relu:activations:0%adv_2_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adv_2_3/MatMul¤
adv_2_3/BiasAdd/ReadVariableOpReadVariableOp'adv_2_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
adv_2_3/BiasAdd/ReadVariableOp¡
adv_2_3/BiasAddBiasAddadv_2_3/MatMul:product:0&adv_2_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adv_2_3/BiasAdd¥
val_2_3/MatMul/ReadVariableOpReadVariableOp&val_2_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
val_2_3/MatMul/ReadVariableOp
val_2_3/MatMulMatMulval_1_3/Relu:activations:0%val_2_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
val_2_3/MatMul¤
val_2_3/BiasAdd/ReadVariableOpReadVariableOp'val_2_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
val_2_3/BiasAdd/ReadVariableOp¡
val_2_3/BiasAddBiasAddval_2_3/MatMul:product:0&val_2_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
val_2_3/BiasAdd«
tf.__operators__.add_3/AddV2AddV2val_2_3/BiasAdd:output:0adv_2_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_3/AddV2
,tf.math.reduce_mean_3/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,tf.math.reduce_mean_3/Mean/reduction_indicesÔ
tf.math.reduce_mean_3/MeanMeanadv_2_3/BiasAdd:output:05tf.math.reduce_mean_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
tf.math.reduce_mean_3/Mean°
tf.math.subtract_3/SubSub tf.__operators__.add_3/AddV2:z:0#tf.math.reduce_mean_3/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.subtract_3/Subµ
IdentityIdentitytf.math.subtract_3/Sub:z:0^adv_0_3/BiasAdd/ReadVariableOp^adv_0_3/MatMul/ReadVariableOp^adv_1_3/BiasAdd/ReadVariableOp^adv_1_3/MatMul/ReadVariableOp^adv_2_3/BiasAdd/ReadVariableOp^adv_2_3/MatMul/ReadVariableOp^dense_0/BiasAdd/ReadVariableOp^dense_0/MatMul/ReadVariableOp^val_0_3/BiasAdd/ReadVariableOp^val_0_3/MatMul/ReadVariableOp^val_1_3/BiasAdd/ReadVariableOp^val_1_3/MatMul/ReadVariableOp^val_2_3/BiasAdd/ReadVariableOp^val_2_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ?: : : : : : : : : : : : : : 2@
adv_0_3/BiasAdd/ReadVariableOpadv_0_3/BiasAdd/ReadVariableOp2>
adv_0_3/MatMul/ReadVariableOpadv_0_3/MatMul/ReadVariableOp2@
adv_1_3/BiasAdd/ReadVariableOpadv_1_3/BiasAdd/ReadVariableOp2>
adv_1_3/MatMul/ReadVariableOpadv_1_3/MatMul/ReadVariableOp2@
adv_2_3/BiasAdd/ReadVariableOpadv_2_3/BiasAdd/ReadVariableOp2>
adv_2_3/MatMul/ReadVariableOpadv_2_3/MatMul/ReadVariableOp2@
dense_0/BiasAdd/ReadVariableOpdense_0/BiasAdd/ReadVariableOp2>
dense_0/MatMul/ReadVariableOpdense_0/MatMul/ReadVariableOp2@
val_0_3/BiasAdd/ReadVariableOpval_0_3/BiasAdd/ReadVariableOp2>
val_0_3/MatMul/ReadVariableOpval_0_3/MatMul/ReadVariableOp2@
val_1_3/BiasAdd/ReadVariableOpval_1_3/BiasAdd/ReadVariableOp2>
val_1_3/MatMul/ReadVariableOpval_1_3/MatMul/ReadVariableOp2@
val_2_3/BiasAdd/ReadVariableOpval_2_3/BiasAdd/ReadVariableOp2>
val_2_3/MatMul/ReadVariableOpval_2_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?
 
_user_specified_nameinputs

ä
'__inference_signature_wrapper_263224507
observation
unknown:	?
	unknown_0:	
	unknown_1:	@
	unknown_2:@
	unknown_3:	@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10:

unknown_11: 

unknown_12:
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference__wrapped_model_2632238602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ?: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?
%
_user_specified_nameobservation


Ü
+__inference_val_1_3_layer_call_fn_263224815

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¬F
â

\__inference_bootstrapped_ddqn_head_3_of_4_layer_call_and_return_conditional_losses_263224411
observation9
&dense_0_matmul_readvariableop_resource:	?6
'dense_0_biasadd_readvariableop_resource:	9
&adv_0_3_matmul_readvariableop_resource:	@5
'adv_0_3_biasadd_readvariableop_resource:@9
&val_0_3_matmul_readvariableop_resource:	@5
'val_0_3_biasadd_readvariableop_resource:@8
&adv_1_3_matmul_readvariableop_resource:@ 5
'adv_1_3_biasadd_readvariableop_resource: 8
&val_1_3_matmul_readvariableop_resource:@ 5
'val_1_3_biasadd_readvariableop_resource: 8
&adv_2_3_matmul_readvariableop_resource: 5
'adv_2_3_biasadd_readvariableop_resource:8
&val_2_3_matmul_readvariableop_resource: 5
'val_2_3_biasadd_readvariableop_resource:
identity¢adv_0_3/BiasAdd/ReadVariableOp¢adv_0_3/MatMul/ReadVariableOp¢adv_1_3/BiasAdd/ReadVariableOp¢adv_1_3/MatMul/ReadVariableOp¢adv_2_3/BiasAdd/ReadVariableOp¢adv_2_3/MatMul/ReadVariableOp¢dense_0/BiasAdd/ReadVariableOp¢dense_0/MatMul/ReadVariableOp¢val_0_3/BiasAdd/ReadVariableOp¢val_0_3/MatMul/ReadVariableOp¢val_1_3/BiasAdd/ReadVariableOp¢val_1_3/MatMul/ReadVariableOp¢val_2_3/BiasAdd/ReadVariableOp¢val_2_3/MatMul/ReadVariableOp¦
dense_0/MatMul/ReadVariableOpReadVariableOp&dense_0_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_0/MatMul/ReadVariableOp
dense_0/MatMulMatMulobservation%dense_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_0/MatMul¥
dense_0/BiasAdd/ReadVariableOpReadVariableOp'dense_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_0/BiasAdd/ReadVariableOp¢
dense_0/BiasAddBiasAdddense_0/MatMul:product:0&dense_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_0/BiasAddq
dense_0/ReluReludense_0/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_0/Relu¦
adv_0_3/MatMul/ReadVariableOpReadVariableOp&adv_0_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
adv_0_3/MatMul/ReadVariableOp
adv_0_3/MatMulMatMuldense_0/Relu:activations:0%adv_0_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_0_3/MatMul¤
adv_0_3/BiasAdd/ReadVariableOpReadVariableOp'adv_0_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
adv_0_3/BiasAdd/ReadVariableOp¡
adv_0_3/BiasAddBiasAddadv_0_3/MatMul:product:0&adv_0_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_0_3/BiasAddp
adv_0_3/ReluReluadv_0_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_0_3/Relu¦
val_0_3/MatMul/ReadVariableOpReadVariableOp&val_0_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
val_0_3/MatMul/ReadVariableOp
val_0_3/MatMulMatMuldense_0/Relu:activations:0%val_0_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_0_3/MatMul¤
val_0_3/BiasAdd/ReadVariableOpReadVariableOp'val_0_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
val_0_3/BiasAdd/ReadVariableOp¡
val_0_3/BiasAddBiasAddval_0_3/MatMul:product:0&val_0_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_0_3/BiasAddp
val_0_3/ReluReluval_0_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_0_3/Relu¥
adv_1_3/MatMul/ReadVariableOpReadVariableOp&adv_1_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
adv_1_3/MatMul/ReadVariableOp
adv_1_3/MatMulMatMuladv_0_3/Relu:activations:0%adv_1_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
adv_1_3/MatMul¤
adv_1_3/BiasAdd/ReadVariableOpReadVariableOp'adv_1_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
adv_1_3/BiasAdd/ReadVariableOp¡
adv_1_3/BiasAddBiasAddadv_1_3/MatMul:product:0&adv_1_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
adv_1_3/BiasAddp
adv_1_3/ReluReluadv_1_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
adv_1_3/Relu¥
val_1_3/MatMul/ReadVariableOpReadVariableOp&val_1_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
val_1_3/MatMul/ReadVariableOp
val_1_3/MatMulMatMulval_0_3/Relu:activations:0%val_1_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
val_1_3/MatMul¤
val_1_3/BiasAdd/ReadVariableOpReadVariableOp'val_1_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
val_1_3/BiasAdd/ReadVariableOp¡
val_1_3/BiasAddBiasAddval_1_3/MatMul:product:0&val_1_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
val_1_3/BiasAddp
val_1_3/ReluReluval_1_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
val_1_3/Relu¥
adv_2_3/MatMul/ReadVariableOpReadVariableOp&adv_2_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
adv_2_3/MatMul/ReadVariableOp
adv_2_3/MatMulMatMuladv_1_3/Relu:activations:0%adv_2_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adv_2_3/MatMul¤
adv_2_3/BiasAdd/ReadVariableOpReadVariableOp'adv_2_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
adv_2_3/BiasAdd/ReadVariableOp¡
adv_2_3/BiasAddBiasAddadv_2_3/MatMul:product:0&adv_2_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adv_2_3/BiasAdd¥
val_2_3/MatMul/ReadVariableOpReadVariableOp&val_2_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
val_2_3/MatMul/ReadVariableOp
val_2_3/MatMulMatMulval_1_3/Relu:activations:0%val_2_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
val_2_3/MatMul¤
val_2_3/BiasAdd/ReadVariableOpReadVariableOp'val_2_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
val_2_3/BiasAdd/ReadVariableOp¡
val_2_3/BiasAddBiasAddval_2_3/MatMul:product:0&val_2_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
val_2_3/BiasAdd«
tf.__operators__.add_3/AddV2AddV2val_2_3/BiasAdd:output:0adv_2_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_3/AddV2
,tf.math.reduce_mean_3/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,tf.math.reduce_mean_3/Mean/reduction_indicesÔ
tf.math.reduce_mean_3/MeanMeanadv_2_3/BiasAdd:output:05tf.math.reduce_mean_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
tf.math.reduce_mean_3/Mean°
tf.math.subtract_3/SubSub tf.__operators__.add_3/AddV2:z:0#tf.math.reduce_mean_3/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.subtract_3/Subµ
IdentityIdentitytf.math.subtract_3/Sub:z:0^adv_0_3/BiasAdd/ReadVariableOp^adv_0_3/MatMul/ReadVariableOp^adv_1_3/BiasAdd/ReadVariableOp^adv_1_3/MatMul/ReadVariableOp^adv_2_3/BiasAdd/ReadVariableOp^adv_2_3/MatMul/ReadVariableOp^dense_0/BiasAdd/ReadVariableOp^dense_0/MatMul/ReadVariableOp^val_0_3/BiasAdd/ReadVariableOp^val_0_3/MatMul/ReadVariableOp^val_1_3/BiasAdd/ReadVariableOp^val_1_3/MatMul/ReadVariableOp^val_2_3/BiasAdd/ReadVariableOp^val_2_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ?: : : : : : : : : : : : : : 2@
adv_0_3/BiasAdd/ReadVariableOpadv_0_3/BiasAdd/ReadVariableOp2>
adv_0_3/MatMul/ReadVariableOpadv_0_3/MatMul/ReadVariableOp2@
adv_1_3/BiasAdd/ReadVariableOpadv_1_3/BiasAdd/ReadVariableOp2>
adv_1_3/MatMul/ReadVariableOpadv_1_3/MatMul/ReadVariableOp2@
adv_2_3/BiasAdd/ReadVariableOpadv_2_3/BiasAdd/ReadVariableOp2>
adv_2_3/MatMul/ReadVariableOpadv_2_3/MatMul/ReadVariableOp2@
dense_0/BiasAdd/ReadVariableOpdense_0/BiasAdd/ReadVariableOp2>
dense_0/MatMul/ReadVariableOpdense_0/MatMul/ReadVariableOp2@
val_0_3/BiasAdd/ReadVariableOpval_0_3/BiasAdd/ReadVariableOp2>
val_0_3/MatMul/ReadVariableOpval_0_3/MatMul/ReadVariableOp2@
val_1_3/BiasAdd/ReadVariableOpval_1_3/BiasAdd/ReadVariableOp2>
val_1_3/MatMul/ReadVariableOpval_1_3/MatMul/ReadVariableOp2@
val_2_3/BiasAdd/ReadVariableOpval_2_3/BiasAdd/ReadVariableOp2>
val_2_3/MatMul/ReadVariableOpval_2_3/MatMul/ReadVariableOp:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?
%
_user_specified_nameobservation"Ì-
saver_filename:0
Identity:0Identity_508"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*½
serving_default©
C
observation4
serving_default_observation:0ÿÿÿÿÿÿÿÿÿ?F
tf.math.subtract_30
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:â¦
Z
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer-8

layer-9
layer-10
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
_default_save_signature
+&call_and_return_all_conditional_losses
__call__"V
_tf_keras_networkèU{"name": "bootstrapped_ddqn_head_3_of_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "bootstrapped_ddqn_head_3_of_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 63]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "observation"}, "name": "observation", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_0", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_0", "inbound_nodes": [[["observation", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "val_0_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "val_0_3", "inbound_nodes": [[["dense_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "adv_0_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "adv_0_3", "inbound_nodes": [[["dense_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "val_1_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "val_1_3", "inbound_nodes": [[["val_0_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "adv_1_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "adv_1_3", "inbound_nodes": [[["adv_0_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "val_2_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "val_2_3", "inbound_nodes": [[["val_1_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "adv_2_3", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "adv_2_3", "inbound_nodes": [[["adv_1_3", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_3", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_3", "inbound_nodes": [["val_2_3", 0, 0, {"y": ["adv_2_3", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean_3", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean_3", "inbound_nodes": [["adv_2_3", 0, 0, {"axis": 1, "keepdims": true}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_3", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract_3", "inbound_nodes": [["tf.__operators__.add_3", 0, 0, {"y": ["tf.math.reduce_mean_3", 0, 0], "name": null}]]}], "input_layers": [["observation", 0, 0]], "output_layers": [["tf.math.subtract_3", 0, 0]]}, "shared_object_id": 25, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 63]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 63]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 63]}, "float32", "observation"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "bootstrapped_ddqn_head_3_of_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 63]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "observation"}, "name": "observation", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_0", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_0", "inbound_nodes": [[["observation", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Dense", "config": {"name": "val_0_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "val_0_3", "inbound_nodes": [[["dense_0", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "adv_0_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "adv_0_3", "inbound_nodes": [[["dense_0", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dense", "config": {"name": "val_1_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "val_1_3", "inbound_nodes": [[["val_0_3", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Dense", "config": {"name": "adv_1_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "adv_1_3", "inbound_nodes": [[["adv_0_3", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "Dense", "config": {"name": "val_2_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "val_2_3", "inbound_nodes": [[["val_1_3", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "Dense", "config": {"name": "adv_2_3", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "adv_2_3", "inbound_nodes": [[["adv_1_3", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_3", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_3", "inbound_nodes": [["val_2_3", 0, 0, {"y": ["adv_2_3", 0, 0], "name": null}]], "shared_object_id": 22}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean_3", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean_3", "inbound_nodes": [["adv_2_3", 0, 0, {"axis": 1, "keepdims": true}]], "shared_object_id": 23}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_3", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract_3", "inbound_nodes": [["tf.__operators__.add_3", 0, 0, {"y": ["tf.math.reduce_mean_3", 0, 0], "name": null}]], "shared_object_id": 24}], "input_layers": [["observation", 0, 0]], "output_layers": [["tf.math.subtract_3", 0, 0]]}}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}, "shared_object_id": 27}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ó"ð
_tf_keras_input_layerÐ{"class_name": "InputLayer", "name": "observation", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 63]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 63]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "observation"}}
	

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ù
_tf_keras_layer¿{"name": "dense_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_0", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["observation", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 63}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 63]}}
ý

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ö
_tf_keras_layer¼{"name": "val_0_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "val_0_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_0", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 29}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
ý

kernel
bias
 trainable_variables
!regularization_losses
"	variables
#	keras_api
+&call_and_return_all_conditional_losses
__call__"Ö
_tf_keras_layer¼{"name": "adv_0_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "adv_0_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_0", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 30}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
þ

$kernel
%bias
&trainable_variables
'regularization_losses
(	variables
)	keras_api
+&call_and_return_all_conditional_losses
__call__"×
_tf_keras_layer½{"name": "val_1_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "val_1_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["val_0_3", 0, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
þ

*kernel
+bias
,trainable_variables
-regularization_losses
.	variables
/	keras_api
+&call_and_return_all_conditional_losses
__call__"×
_tf_keras_layer½{"name": "adv_1_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "adv_1_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["adv_0_3", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 32}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ÿ

0kernel
1bias
2trainable_variables
3regularization_losses
4	variables
5	keras_api
+&call_and_return_all_conditional_losses
__call__"Ø
_tf_keras_layer¾{"name": "val_2_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "val_2_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["val_1_3", 0, 0, {}]]], "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
ÿ

6kernel
7bias
8trainable_variables
9regularization_losses
:	variables
;	keras_api
+&call_and_return_all_conditional_losses
__call__"Ø
_tf_keras_layer¾{"name": "adv_2_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "adv_2_3", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["adv_1_3", 0, 0, {}]]], "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 34}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
Ü
<	keras_api"Ê
_tf_keras_layer°{"name": "tf.__operators__.add_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_3", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "inbound_nodes": [["val_2_3", 0, 0, {"y": ["adv_2_3", 0, 0], "name": null}]], "shared_object_id": 22}
Ð
=	keras_api"¾
_tf_keras_layer¤{"name": "tf.math.reduce_mean_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean_3", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "inbound_nodes": [["adv_2_3", 0, 0, {"axis": 1, "keepdims": true}]], "shared_object_id": 23}
í
>	keras_api"Û
_tf_keras_layerÁ{"name": "tf.math.subtract_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_3", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "inbound_nodes": [["tf.__operators__.add_3", 0, 0, {"y": ["tf.math.reduce_mean_3", 0, 0], "name": null}]], "shared_object_id": 24}
Ü
?iter

@beta_1

Abeta_2
	Bdecay
Clearning_ratemqmrmsmtmumv$mw%mx*my+mz0m{1m|6m}7m~vvvvvv$v%v*v+v0v1v6v7v"
	optimizer

0
1
2
3
4
5
$6
%7
*8
+9
010
111
612
713"
trackable_list_wrapper
 "
trackable_list_wrapper

0
1
2
3
4
5
$6
%7
*8
+9
010
111
612
713"
trackable_list_wrapper
Î
trainable_variables
regularization_losses
Dnon_trainable_variables
Emetrics

Flayers
Glayer_metrics
	variables
Hlayer_regularization_losses
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
!:	?2dense_0/kernel
:2dense_0/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
trainable_variables
regularization_losses
Inon_trainable_variables
Jmetrics

Klayers
Llayer_metrics
	variables
Mlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:	@2val_0_3/kernel
:@2val_0_3/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
trainable_variables
regularization_losses
Nnon_trainable_variables
Ometrics

Players
Qlayer_metrics
	variables
Rlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:	@2adv_0_3/kernel
:@2adv_0_3/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
 trainable_variables
!regularization_losses
Snon_trainable_variables
Tmetrics

Ulayers
Vlayer_metrics
"	variables
Wlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 :@ 2val_1_3/kernel
: 2val_1_3/bias
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
°
&trainable_variables
'regularization_losses
Xnon_trainable_variables
Ymetrics

Zlayers
[layer_metrics
(	variables
\layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 :@ 2adv_1_3/kernel
: 2adv_1_3/bias
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
°
,trainable_variables
-regularization_losses
]non_trainable_variables
^metrics

_layers
`layer_metrics
.	variables
alayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 : 2val_2_3/kernel
:2val_2_3/bias
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
°
2trainable_variables
3regularization_losses
bnon_trainable_variables
cmetrics

dlayers
elayer_metrics
4	variables
flayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 : 2adv_2_3/kernel
:2adv_2_3/bias
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
°
8trainable_variables
9regularization_losses
gnon_trainable_variables
hmetrics

ilayers
jlayer_metrics
:	variables
klayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
'
l0"
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
Ô
	mtotal
	ncount
o	variables
p	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 35}
:  (2total
:  (2count
.
m0
n1"
trackable_list_wrapper
-
o	variables"
_generic_user_object
&:$	?2Adam/dense_0/kernel/m
 :2Adam/dense_0/bias/m
&:$	@2Adam/val_0_3/kernel/m
:@2Adam/val_0_3/bias/m
&:$	@2Adam/adv_0_3/kernel/m
:@2Adam/adv_0_3/bias/m
%:#@ 2Adam/val_1_3/kernel/m
: 2Adam/val_1_3/bias/m
%:#@ 2Adam/adv_1_3/kernel/m
: 2Adam/adv_1_3/bias/m
%:# 2Adam/val_2_3/kernel/m
:2Adam/val_2_3/bias/m
%:# 2Adam/adv_2_3/kernel/m
:2Adam/adv_2_3/bias/m
&:$	?2Adam/dense_0/kernel/v
 :2Adam/dense_0/bias/v
&:$	@2Adam/val_0_3/kernel/v
:@2Adam/val_0_3/bias/v
&:$	@2Adam/adv_0_3/kernel/v
:@2Adam/adv_0_3/bias/v
%:#@ 2Adam/val_1_3/kernel/v
: 2Adam/val_1_3/bias/v
%:#@ 2Adam/adv_1_3/kernel/v
: 2Adam/adv_1_3/bias/v
%:# 2Adam/val_2_3/kernel/v
:2Adam/val_2_3/bias/v
%:# 2Adam/adv_2_3/kernel/v
:2Adam/adv_2_3/bias/v
æ2ã
$__inference__wrapped_model_263223860º
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª **¢'
%"
observationÿÿÿÿÿÿÿÿÿ?
¾2»
\__inference_bootstrapped_ddqn_head_3_of_4_layer_call_and_return_conditional_losses_263224562
\__inference_bootstrapped_ddqn_head_3_of_4_layer_call_and_return_conditional_losses_263224617
\__inference_bootstrapped_ddqn_head_3_of_4_layer_call_and_return_conditional_losses_263224411
\__inference_bootstrapped_ddqn_head_3_of_4_layer_call_and_return_conditional_losses_263224466À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
A__inference_bootstrapped_ddqn_head_3_of_4_layer_call_fn_263223916
A__inference_bootstrapped_ddqn_head_3_of_4_layer_call_fn_263224672
A__inference_bootstrapped_ddqn_head_3_of_4_layer_call_fn_263224727
A__inference_bootstrapped_ddqn_head_3_of_4_layer_call_fn_263224356À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ð2í
F__inference_dense_0_layer_call_and_return_conditional_losses_263224738¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_0_layer_call_fn_263224749¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_val_0_3_layer_call_and_return_conditional_losses_263224760¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_val_0_3_layer_call_fn_263224771¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_adv_0_3_layer_call_and_return_conditional_losses_263224782¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_adv_0_3_layer_call_fn_263224793¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_val_1_3_layer_call_and_return_conditional_losses_263224804¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_val_1_3_layer_call_fn_263224815¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_adv_1_3_layer_call_and_return_conditional_losses_263224826¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_adv_1_3_layer_call_fn_263224837¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_val_2_3_layer_call_and_return_conditional_losses_263224847¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_val_2_3_layer_call_fn_263224857¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_adv_2_3_layer_call_and_return_conditional_losses_263224867¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_adv_2_3_layer_call_fn_263224877¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÒBÏ
'__inference_signature_wrapper_263224507observation"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ¸
$__inference__wrapped_model_263223860*+$%67014¢1
*¢'
%"
observationÿÿÿÿÿÿÿÿÿ?
ª "GªD
B
tf.math.subtract_3,)
tf.math.subtract_3ÿÿÿÿÿÿÿÿÿ§
F__inference_adv_0_3_layer_call_and_return_conditional_losses_263224782]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
+__inference_adv_0_3_layer_call_fn_263224793P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@¦
F__inference_adv_1_3_layer_call_and_return_conditional_losses_263224826\*+/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 ~
+__inference_adv_1_3_layer_call_fn_263224837O*+/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ ¦
F__inference_adv_2_3_layer_call_and_return_conditional_losses_263224867\67/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_adv_2_3_layer_call_fn_263224877O67/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿÕ
\__inference_bootstrapped_ddqn_head_3_of_4_layer_call_and_return_conditional_losses_263224411u*+$%6701<¢9
2¢/
%"
observationÿÿÿÿÿÿÿÿÿ?
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Õ
\__inference_bootstrapped_ddqn_head_3_of_4_layer_call_and_return_conditional_losses_263224466u*+$%6701<¢9
2¢/
%"
observationÿÿÿÿÿÿÿÿÿ?
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ð
\__inference_bootstrapped_ddqn_head_3_of_4_layer_call_and_return_conditional_losses_263224562p*+$%67017¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ?
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ð
\__inference_bootstrapped_ddqn_head_3_of_4_layer_call_and_return_conditional_losses_263224617p*+$%67017¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ?
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ­
A__inference_bootstrapped_ddqn_head_3_of_4_layer_call_fn_263223916h*+$%6701<¢9
2¢/
%"
observationÿÿÿÿÿÿÿÿÿ?
p 

 
ª "ÿÿÿÿÿÿÿÿÿ­
A__inference_bootstrapped_ddqn_head_3_of_4_layer_call_fn_263224356h*+$%6701<¢9
2¢/
%"
observationÿÿÿÿÿÿÿÿÿ?
p

 
ª "ÿÿÿÿÿÿÿÿÿ¨
A__inference_bootstrapped_ddqn_head_3_of_4_layer_call_fn_263224672c*+$%67017¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ?
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¨
A__inference_bootstrapped_ddqn_head_3_of_4_layer_call_fn_263224727c*+$%67017¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ?
p

 
ª "ÿÿÿÿÿÿÿÿÿ§
F__inference_dense_0_layer_call_and_return_conditional_losses_263224738]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ?
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dense_0_layer_call_fn_263224749P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ?
ª "ÿÿÿÿÿÿÿÿÿÊ
'__inference_signature_wrapper_263224507*+$%6701C¢@
¢ 
9ª6
4
observation%"
observationÿÿÿÿÿÿÿÿÿ?"GªD
B
tf.math.subtract_3,)
tf.math.subtract_3ÿÿÿÿÿÿÿÿÿ§
F__inference_val_0_3_layer_call_and_return_conditional_losses_263224760]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
+__inference_val_0_3_layer_call_fn_263224771P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@¦
F__inference_val_1_3_layer_call_and_return_conditional_losses_263224804\$%/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 ~
+__inference_val_1_3_layer_call_fn_263224815O$%/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ ¦
F__inference_val_2_3_layer_call_and_return_conditional_losses_263224847\01/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_val_2_3_layer_call_fn_263224857O01/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ