ܡ
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
?
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
delete_old_dirsbool(?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ?
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	??*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	??*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?*
dtype0
u
val_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*
shared_nameval_1/kernel
n
 val_1/kernel/Read/ReadVariableOpReadVariableOpval_1/kernel*
_output_shapes
:	?@*
dtype0
l

val_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
val_1/bias
e
val_1/bias/Read/ReadVariableOpReadVariableOp
val_1/bias*
_output_shapes
:@*
dtype0
u
adv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*
shared_nameadv_1/kernel
n
 adv_1/kernel/Read/ReadVariableOpReadVariableOpadv_1/kernel*
_output_shapes
:	?@*
dtype0
l

adv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
adv_1/bias
e
adv_1/bias/Read/ReadVariableOpReadVariableOp
adv_1/bias*
_output_shapes
:@*
dtype0
t
val_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_nameval_2/kernel
m
 val_2/kernel/Read/ReadVariableOpReadVariableOpval_2/kernel*
_output_shapes

:@@*
dtype0
l

val_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
val_2/bias
e
val_2/bias/Read/ReadVariableOpReadVariableOp
val_2/bias*
_output_shapes
:@*
dtype0
t
adv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_nameadv_2/kernel
m
 adv_2/kernel/Read/ReadVariableOpReadVariableOpadv_2/kernel*
_output_shapes

:@@*
dtype0
l

adv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
adv_2/bias
e
adv_2/bias/Read/ReadVariableOpReadVariableOp
adv_2/bias*
_output_shapes
:@*
dtype0
t
val_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_nameval_3/kernel
m
 val_3/kernel/Read/ReadVariableOpReadVariableOpval_3/kernel*
_output_shapes

:@*
dtype0
l

val_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
val_3/bias
e
val_3/bias/Read/ReadVariableOpReadVariableOp
val_3/bias*
_output_shapes
:*
dtype0
t
adv_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_nameadv_3/kernel
m
 adv_3/kernel/Read/ReadVariableOpReadVariableOpadv_3/kernel*
_output_shapes

:@*
dtype0
l

adv_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
adv_3/bias
e
adv_3/bias/Read/ReadVariableOpReadVariableOp
adv_3/bias*
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
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	??*&
shared_nameAdam/dense_1/kernel/m
?
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	??*
dtype0

Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_1/bias/m
x
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/val_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*$
shared_nameAdam/val_1/kernel/m
|
'Adam/val_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/val_1/kernel/m*
_output_shapes
:	?@*
dtype0
z
Adam/val_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/val_1/bias/m
s
%Adam/val_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/val_1/bias/m*
_output_shapes
:@*
dtype0
?
Adam/adv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*$
shared_nameAdam/adv_1/kernel/m
|
'Adam/adv_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/adv_1/kernel/m*
_output_shapes
:	?@*
dtype0
z
Adam/adv_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/adv_1/bias/m
s
%Adam/adv_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/adv_1/bias/m*
_output_shapes
:@*
dtype0
?
Adam/val_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*$
shared_nameAdam/val_2/kernel/m
{
'Adam/val_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/val_2/kernel/m*
_output_shapes

:@@*
dtype0
z
Adam/val_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/val_2/bias/m
s
%Adam/val_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/val_2/bias/m*
_output_shapes
:@*
dtype0
?
Adam/adv_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*$
shared_nameAdam/adv_2/kernel/m
{
'Adam/adv_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/adv_2/kernel/m*
_output_shapes

:@@*
dtype0
z
Adam/adv_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/adv_2/bias/m
s
%Adam/adv_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/adv_2/bias/m*
_output_shapes
:@*
dtype0
?
Adam/val_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*$
shared_nameAdam/val_3/kernel/m
{
'Adam/val_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/val_3/kernel/m*
_output_shapes

:@*
dtype0
z
Adam/val_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/val_3/bias/m
s
%Adam/val_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/val_3/bias/m*
_output_shapes
:*
dtype0
?
Adam/adv_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*$
shared_nameAdam/adv_3/kernel/m
{
'Adam/adv_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/adv_3/kernel/m*
_output_shapes

:@*
dtype0
z
Adam/adv_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/adv_3/bias/m
s
%Adam/adv_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/adv_3/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	??*&
shared_nameAdam/dense_1/kernel/v
?
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	??*
dtype0

Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_1/bias/v
x
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/val_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*$
shared_nameAdam/val_1/kernel/v
|
'Adam/val_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/val_1/kernel/v*
_output_shapes
:	?@*
dtype0
z
Adam/val_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/val_1/bias/v
s
%Adam/val_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/val_1/bias/v*
_output_shapes
:@*
dtype0
?
Adam/adv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*$
shared_nameAdam/adv_1/kernel/v
|
'Adam/adv_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/adv_1/kernel/v*
_output_shapes
:	?@*
dtype0
z
Adam/adv_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/adv_1/bias/v
s
%Adam/adv_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/adv_1/bias/v*
_output_shapes
:@*
dtype0
?
Adam/val_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*$
shared_nameAdam/val_2/kernel/v
{
'Adam/val_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/val_2/kernel/v*
_output_shapes

:@@*
dtype0
z
Adam/val_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/val_2/bias/v
s
%Adam/val_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/val_2/bias/v*
_output_shapes
:@*
dtype0
?
Adam/adv_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*$
shared_nameAdam/adv_2/kernel/v
{
'Adam/adv_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/adv_2/kernel/v*
_output_shapes

:@@*
dtype0
z
Adam/adv_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/adv_2/bias/v
s
%Adam/adv_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/adv_2/bias/v*
_output_shapes
:@*
dtype0
?
Adam/val_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*$
shared_nameAdam/val_3/kernel/v
{
'Adam/val_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/val_3/kernel/v*
_output_shapes

:@*
dtype0
z
Adam/val_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/val_3/bias/v
s
%Adam/val_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/val_3/bias/v*
_output_shapes
:*
dtype0
?
Adam/adv_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*$
shared_nameAdam/adv_3/kernel/v
{
'Adam/adv_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/adv_3/kernel/v*
_output_shapes

:@*
dtype0
z
Adam/adv_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/adv_3/bias/v
s
%Adam/adv_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/adv_3/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?F
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?E
value?EB?E B?E
?
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
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
h

$kernel
%bias
&	variables
'regularization_losses
(trainable_variables
)	keras_api
h

*kernel
+bias
,	variables
-regularization_losses
.trainable_variables
/	keras_api
h

0kernel
1bias
2	variables
3regularization_losses
4trainable_variables
5	keras_api
h

6kernel
7bias
8	variables
9regularization_losses
:trainable_variables
;	keras_api

<	keras_api

=	keras_api

>	keras_api
?
?iter

@beta_1

Abeta_2
	Bdecay
Clearning_ratemqmrmsmtmumv$mw%mx*my+mz0m{1m|6m}7m~vv?v?v?v?v?$v?%v?*v?+v?0v?1v?6v?7v?
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
?
Dnon_trainable_variables

Elayers
Flayer_metrics
Glayer_regularization_losses
	variables
regularization_losses
trainable_variables
Hmetrics
 
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
Inon_trainable_variables

Jlayers
Klayer_metrics
Llayer_regularization_losses
	variables
regularization_losses
trainable_variables
Mmetrics
XV
VARIABLE_VALUEval_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
val_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
Nnon_trainable_variables

Olayers
Player_metrics
Qlayer_regularization_losses
	variables
regularization_losses
trainable_variables
Rmetrics
XV
VARIABLE_VALUEadv_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
adv_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
Snon_trainable_variables

Tlayers
Ulayer_metrics
Vlayer_regularization_losses
 	variables
!regularization_losses
"trainable_variables
Wmetrics
XV
VARIABLE_VALUEval_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
val_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
 

$0
%1
?
Xnon_trainable_variables

Ylayers
Zlayer_metrics
[layer_regularization_losses
&	variables
'regularization_losses
(trainable_variables
\metrics
XV
VARIABLE_VALUEadv_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
adv_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1
 

*0
+1
?
]non_trainable_variables

^layers
_layer_metrics
`layer_regularization_losses
,	variables
-regularization_losses
.trainable_variables
ametrics
XV
VARIABLE_VALUEval_3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
val_3/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11
 

00
11
?
bnon_trainable_variables

clayers
dlayer_metrics
elayer_regularization_losses
2	variables
3regularization_losses
4trainable_variables
fmetrics
XV
VARIABLE_VALUEadv_3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
adv_3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71
 

60
71
?
gnon_trainable_variables

hlayers
ilayer_metrics
jlayer_regularization_losses
8	variables
9regularization_losses
:trainable_variables
kmetrics
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

l0
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
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/val_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/val_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/adv_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/adv_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/val_2/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/val_2/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/adv_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/adv_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/val_3/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/val_3/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/adv_3/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/adv_3/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/val_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/val_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/adv_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/adv_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/val_2/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/val_2/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/adv_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/adv_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/val_3/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/val_3/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/adv_3/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/adv_3/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
serving_default_observationPlaceholder*'
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_observationdense_1/kerneldense_1/biasadv_1/kernel
adv_1/biasval_1/kernel
val_1/biasadv_2/kernel
adv_2/biasval_2/kernel
val_2/biasadv_3/kernel
adv_3/biasval_3/kernel
val_3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_28344956
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp val_1/kernel/Read/ReadVariableOpval_1/bias/Read/ReadVariableOp adv_1/kernel/Read/ReadVariableOpadv_1/bias/Read/ReadVariableOp val_2/kernel/Read/ReadVariableOpval_2/bias/Read/ReadVariableOp adv_2/kernel/Read/ReadVariableOpadv_2/bias/Read/ReadVariableOp val_3/kernel/Read/ReadVariableOpval_3/bias/Read/ReadVariableOp adv_3/kernel/Read/ReadVariableOpadv_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp'Adam/val_1/kernel/m/Read/ReadVariableOp%Adam/val_1/bias/m/Read/ReadVariableOp'Adam/adv_1/kernel/m/Read/ReadVariableOp%Adam/adv_1/bias/m/Read/ReadVariableOp'Adam/val_2/kernel/m/Read/ReadVariableOp%Adam/val_2/bias/m/Read/ReadVariableOp'Adam/adv_2/kernel/m/Read/ReadVariableOp%Adam/adv_2/bias/m/Read/ReadVariableOp'Adam/val_3/kernel/m/Read/ReadVariableOp%Adam/val_3/bias/m/Read/ReadVariableOp'Adam/adv_3/kernel/m/Read/ReadVariableOp%Adam/adv_3/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp'Adam/val_1/kernel/v/Read/ReadVariableOp%Adam/val_1/bias/v/Read/ReadVariableOp'Adam/adv_1/kernel/v/Read/ReadVariableOp%Adam/adv_1/bias/v/Read/ReadVariableOp'Adam/val_2/kernel/v/Read/ReadVariableOp%Adam/val_2/bias/v/Read/ReadVariableOp'Adam/adv_2/kernel/v/Read/ReadVariableOp%Adam/adv_2/bias/v/Read/ReadVariableOp'Adam/val_3/kernel/v/Read/ReadVariableOp%Adam/val_3/bias/v/Read/ReadVariableOp'Adam/adv_3/kernel/v/Read/ReadVariableOp%Adam/adv_3/bias/v/Read/ReadVariableOpConst*>
Tin7
523	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_save_28345440
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1/kerneldense_1/biasval_1/kernel
val_1/biasadv_1/kernel
adv_1/biasval_2/kernel
val_2/biasadv_2/kernel
adv_2/biasval_3/kernel
val_3/biasadv_3/kernel
adv_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/val_1/kernel/mAdam/val_1/bias/mAdam/adv_1/kernel/mAdam/adv_1/bias/mAdam/val_2/kernel/mAdam/val_2/bias/mAdam/adv_2/kernel/mAdam/adv_2/bias/mAdam/val_3/kernel/mAdam/val_3/bias/mAdam/adv_3/kernel/mAdam/adv_3/bias/mAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/val_1/kernel/vAdam/val_1/bias/vAdam/adv_1/kernel/vAdam/adv_1/bias/vAdam/val_2/kernel/vAdam/val_2/bias/vAdam/adv_2/kernel/vAdam/adv_2/bias/vAdam/val_3/kernel/vAdam/val_3/bias/vAdam/adv_3/kernel/vAdam/adv_3/bias/v*=
Tin6
422*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference__traced_restore_28345597ح
?

?
C__inference_adv_2_layer_call_and_return_conditional_losses_28345223

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
E__inference_dense_1_layer_call_and_return_conditional_losses_28345143

inputs1
matmul_readvariableop_resource:	??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_28344956
observation
unknown:	??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:	?@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:

unknown_11:@

unknown_12:
identity??StatefulPartitionedCall?
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
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__wrapped_model_283444572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:??????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:??????????
%
_user_specified_nameobservation
?
?
(__inference_val_2_layer_call_fn_28345212

inputs
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_val_2_layer_call_and_return_conditional_losses_283445432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_adv_3_layer_call_fn_28345270

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_adv_3_layer_call_and_return_conditional_losses_283445592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?+
?
C__inference_model_layer_call_and_return_conditional_losses_28344765

inputs#
dense_1_28344725:	??
dense_1_28344727:	?!
adv_1_28344730:	?@
adv_1_28344732:@!
val_1_28344735:	?@
val_1_28344737:@ 
adv_2_28344740:@@
adv_2_28344742:@ 
val_2_28344745:@@
val_2_28344747:@ 
adv_3_28344750:@
adv_3_28344752: 
val_3_28344755:@
val_3_28344757:
identity??adv_1/StatefulPartitionedCall?adv_2/StatefulPartitionedCall?adv_3/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?val_1/StatefulPartitionedCall?val_2/StatefulPartitionedCall?val_3/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_28344725dense_1_28344727*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_283444752!
dense_1/StatefulPartitionedCall?
adv_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0adv_1_28344730adv_1_28344732*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_adv_1_layer_call_and_return_conditional_losses_283444922
adv_1/StatefulPartitionedCall?
val_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0val_1_28344735val_1_28344737*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_val_1_layer_call_and_return_conditional_losses_283445092
val_1/StatefulPartitionedCall?
adv_2/StatefulPartitionedCallStatefulPartitionedCall&adv_1/StatefulPartitionedCall:output:0adv_2_28344740adv_2_28344742*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_adv_2_layer_call_and_return_conditional_losses_283445262
adv_2/StatefulPartitionedCall?
val_2/StatefulPartitionedCallStatefulPartitionedCall&val_1/StatefulPartitionedCall:output:0val_2_28344745val_2_28344747*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_val_2_layer_call_and_return_conditional_losses_283445432
val_2/StatefulPartitionedCall?
adv_3/StatefulPartitionedCallStatefulPartitionedCall&adv_2/StatefulPartitionedCall:output:0adv_3_28344750adv_3_28344752*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_adv_3_layer_call_and_return_conditional_losses_283445592
adv_3/StatefulPartitionedCall?
val_3/StatefulPartitionedCallStatefulPartitionedCall&val_2/StatefulPartitionedCall:output:0val_3_28344755val_3_28344757*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_val_3_layer_call_and_return_conditional_losses_283445752
val_3/StatefulPartitionedCall?
tf.__operators__.add/AddV2AddV2&val_3/StatefulPartitionedCall:output:0&adv_3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add/AddV2?
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_mean/Mean/reduction_indices?
tf.math.reduce_mean/MeanMean&adv_3/StatefulPartitionedCall:output:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
tf.math.reduce_mean/Mean?
tf.math.subtract/SubSubtf.__operators__.add/AddV2:z:0!tf.math.reduce_mean/Mean:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract/Sub?
IdentityIdentitytf.math.subtract/Sub:z:0^adv_1/StatefulPartitionedCall^adv_2/StatefulPartitionedCall^adv_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^val_1/StatefulPartitionedCall^val_2/StatefulPartitionedCall^val_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:??????????: : : : : : : : : : : : : : 2>
adv_1/StatefulPartitionedCalladv_1/StatefulPartitionedCall2>
adv_2/StatefulPartitionedCalladv_2/StatefulPartitionedCall2>
adv_3/StatefulPartitionedCalladv_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
val_1/StatefulPartitionedCallval_1/StatefulPartitionedCall2>
val_2/StatefulPartitionedCallval_2/StatefulPartitionedCall2>
val_3/StatefulPartitionedCallval_3/StatefulPartitionedCall:O K
'
_output_shapes
:??????????
 
_user_specified_nameinputs
?L
?
#__inference__wrapped_model_28344457
observation?
,model_dense_1_matmul_readvariableop_resource:	??<
-model_dense_1_biasadd_readvariableop_resource:	?=
*model_adv_1_matmul_readvariableop_resource:	?@9
+model_adv_1_biasadd_readvariableop_resource:@=
*model_val_1_matmul_readvariableop_resource:	?@9
+model_val_1_biasadd_readvariableop_resource:@<
*model_adv_2_matmul_readvariableop_resource:@@9
+model_adv_2_biasadd_readvariableop_resource:@<
*model_val_2_matmul_readvariableop_resource:@@9
+model_val_2_biasadd_readvariableop_resource:@<
*model_adv_3_matmul_readvariableop_resource:@9
+model_adv_3_biasadd_readvariableop_resource:<
*model_val_3_matmul_readvariableop_resource:@9
+model_val_3_biasadd_readvariableop_resource:
identity??"model/adv_1/BiasAdd/ReadVariableOp?!model/adv_1/MatMul/ReadVariableOp?"model/adv_2/BiasAdd/ReadVariableOp?!model/adv_2/MatMul/ReadVariableOp?"model/adv_3/BiasAdd/ReadVariableOp?!model/adv_3/MatMul/ReadVariableOp?$model/dense_1/BiasAdd/ReadVariableOp?#model/dense_1/MatMul/ReadVariableOp?"model/val_1/BiasAdd/ReadVariableOp?!model/val_1/MatMul/ReadVariableOp?"model/val_2/BiasAdd/ReadVariableOp?!model/val_2/MatMul/ReadVariableOp?"model/val_3/BiasAdd/ReadVariableOp?!model/val_3/MatMul/ReadVariableOp?
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	??*
dtype02%
#model/dense_1/MatMul/ReadVariableOp?
model/dense_1/MatMulMatMulobservation+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/dense_1/MatMul?
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp?
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/dense_1/BiasAdd?
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/dense_1/Relu?
!model/adv_1/MatMul/ReadVariableOpReadVariableOp*model_adv_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02#
!model/adv_1/MatMul/ReadVariableOp?
model/adv_1/MatMulMatMul model/dense_1/Relu:activations:0)model/adv_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/adv_1/MatMul?
"model/adv_1/BiasAdd/ReadVariableOpReadVariableOp+model_adv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"model/adv_1/BiasAdd/ReadVariableOp?
model/adv_1/BiasAddBiasAddmodel/adv_1/MatMul:product:0*model/adv_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/adv_1/BiasAdd|
model/adv_1/ReluRelumodel/adv_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model/adv_1/Relu?
!model/val_1/MatMul/ReadVariableOpReadVariableOp*model_val_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02#
!model/val_1/MatMul/ReadVariableOp?
model/val_1/MatMulMatMul model/dense_1/Relu:activations:0)model/val_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/val_1/MatMul?
"model/val_1/BiasAdd/ReadVariableOpReadVariableOp+model_val_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"model/val_1/BiasAdd/ReadVariableOp?
model/val_1/BiasAddBiasAddmodel/val_1/MatMul:product:0*model/val_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/val_1/BiasAdd|
model/val_1/ReluRelumodel/val_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model/val_1/Relu?
!model/adv_2/MatMul/ReadVariableOpReadVariableOp*model_adv_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02#
!model/adv_2/MatMul/ReadVariableOp?
model/adv_2/MatMulMatMulmodel/adv_1/Relu:activations:0)model/adv_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/adv_2/MatMul?
"model/adv_2/BiasAdd/ReadVariableOpReadVariableOp+model_adv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"model/adv_2/BiasAdd/ReadVariableOp?
model/adv_2/BiasAddBiasAddmodel/adv_2/MatMul:product:0*model/adv_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/adv_2/BiasAdd|
model/adv_2/ReluRelumodel/adv_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model/adv_2/Relu?
!model/val_2/MatMul/ReadVariableOpReadVariableOp*model_val_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02#
!model/val_2/MatMul/ReadVariableOp?
model/val_2/MatMulMatMulmodel/val_1/Relu:activations:0)model/val_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/val_2/MatMul?
"model/val_2/BiasAdd/ReadVariableOpReadVariableOp+model_val_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"model/val_2/BiasAdd/ReadVariableOp?
model/val_2/BiasAddBiasAddmodel/val_2/MatMul:product:0*model/val_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/val_2/BiasAdd|
model/val_2/ReluRelumodel/val_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model/val_2/Relu?
!model/adv_3/MatMul/ReadVariableOpReadVariableOp*model_adv_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02#
!model/adv_3/MatMul/ReadVariableOp?
model/adv_3/MatMulMatMulmodel/adv_2/Relu:activations:0)model/adv_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/adv_3/MatMul?
"model/adv_3/BiasAdd/ReadVariableOpReadVariableOp+model_adv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/adv_3/BiasAdd/ReadVariableOp?
model/adv_3/BiasAddBiasAddmodel/adv_3/MatMul:product:0*model/adv_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/adv_3/BiasAdd?
!model/val_3/MatMul/ReadVariableOpReadVariableOp*model_val_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02#
!model/val_3/MatMul/ReadVariableOp?
model/val_3/MatMulMatMulmodel/val_2/Relu:activations:0)model/val_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/val_3/MatMul?
"model/val_3/BiasAdd/ReadVariableOpReadVariableOp+model_val_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/val_3/BiasAdd/ReadVariableOp?
model/val_3/BiasAddBiasAddmodel/val_3/MatMul:product:0*model/val_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/val_3/BiasAdd?
 model/tf.__operators__.add/AddV2AddV2model/val_3/BiasAdd:output:0model/adv_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2"
 model/tf.__operators__.add/AddV2?
0model/tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :22
0model/tf.math.reduce_mean/Mean/reduction_indices?
model/tf.math.reduce_mean/MeanMeanmodel/adv_3/BiasAdd:output:09model/tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2 
model/tf.math.reduce_mean/Mean?
model/tf.math.subtract/SubSub$model/tf.__operators__.add/AddV2:z:0'model/tf.math.reduce_mean/Mean:output:0*
T0*'
_output_shapes
:?????????2
model/tf.math.subtract/Sub?
IdentityIdentitymodel/tf.math.subtract/Sub:z:0#^model/adv_1/BiasAdd/ReadVariableOp"^model/adv_1/MatMul/ReadVariableOp#^model/adv_2/BiasAdd/ReadVariableOp"^model/adv_2/MatMul/ReadVariableOp#^model/adv_3/BiasAdd/ReadVariableOp"^model/adv_3/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp#^model/val_1/BiasAdd/ReadVariableOp"^model/val_1/MatMul/ReadVariableOp#^model/val_2/BiasAdd/ReadVariableOp"^model/val_2/MatMul/ReadVariableOp#^model/val_3/BiasAdd/ReadVariableOp"^model/val_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:??????????: : : : : : : : : : : : : : 2H
"model/adv_1/BiasAdd/ReadVariableOp"model/adv_1/BiasAdd/ReadVariableOp2F
!model/adv_1/MatMul/ReadVariableOp!model/adv_1/MatMul/ReadVariableOp2H
"model/adv_2/BiasAdd/ReadVariableOp"model/adv_2/BiasAdd/ReadVariableOp2F
!model/adv_2/MatMul/ReadVariableOp!model/adv_2/MatMul/ReadVariableOp2H
"model/adv_3/BiasAdd/ReadVariableOp"model/adv_3/BiasAdd/ReadVariableOp2F
!model/adv_3/MatMul/ReadVariableOp!model/adv_3/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2H
"model/val_1/BiasAdd/ReadVariableOp"model/val_1/BiasAdd/ReadVariableOp2F
!model/val_1/MatMul/ReadVariableOp!model/val_1/MatMul/ReadVariableOp2H
"model/val_2/BiasAdd/ReadVariableOp"model/val_2/BiasAdd/ReadVariableOp2F
!model/val_2/MatMul/ReadVariableOp!model/val_2/MatMul/ReadVariableOp2H
"model/val_3/BiasAdd/ReadVariableOp"model/val_3/BiasAdd/ReadVariableOp2F
!model/val_3/MatMul/ReadVariableOp!model/val_3/MatMul/ReadVariableOp:T P
'
_output_shapes
:??????????
%
_user_specified_nameobservation
?+
?
C__inference_model_layer_call_and_return_conditional_losses_28344915
observation#
dense_1_28344875:	??
dense_1_28344877:	?!
adv_1_28344880:	?@
adv_1_28344882:@!
val_1_28344885:	?@
val_1_28344887:@ 
adv_2_28344890:@@
adv_2_28344892:@ 
val_2_28344895:@@
val_2_28344897:@ 
adv_3_28344900:@
adv_3_28344902: 
val_3_28344905:@
val_3_28344907:
identity??adv_1/StatefulPartitionedCall?adv_2/StatefulPartitionedCall?adv_3/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?val_1/StatefulPartitionedCall?val_2/StatefulPartitionedCall?val_3/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallobservationdense_1_28344875dense_1_28344877*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_283444752!
dense_1/StatefulPartitionedCall?
adv_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0adv_1_28344880adv_1_28344882*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_adv_1_layer_call_and_return_conditional_losses_283444922
adv_1/StatefulPartitionedCall?
val_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0val_1_28344885val_1_28344887*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_val_1_layer_call_and_return_conditional_losses_283445092
val_1/StatefulPartitionedCall?
adv_2/StatefulPartitionedCallStatefulPartitionedCall&adv_1/StatefulPartitionedCall:output:0adv_2_28344890adv_2_28344892*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_adv_2_layer_call_and_return_conditional_losses_283445262
adv_2/StatefulPartitionedCall?
val_2/StatefulPartitionedCallStatefulPartitionedCall&val_1/StatefulPartitionedCall:output:0val_2_28344895val_2_28344897*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_val_2_layer_call_and_return_conditional_losses_283445432
val_2/StatefulPartitionedCall?
adv_3/StatefulPartitionedCallStatefulPartitionedCall&adv_2/StatefulPartitionedCall:output:0adv_3_28344900adv_3_28344902*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_adv_3_layer_call_and_return_conditional_losses_283445592
adv_3/StatefulPartitionedCall?
val_3/StatefulPartitionedCallStatefulPartitionedCall&val_2/StatefulPartitionedCall:output:0val_3_28344905val_3_28344907*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_val_3_layer_call_and_return_conditional_losses_283445752
val_3/StatefulPartitionedCall?
tf.__operators__.add/AddV2AddV2&val_3/StatefulPartitionedCall:output:0&adv_3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add/AddV2?
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_mean/Mean/reduction_indices?
tf.math.reduce_mean/MeanMean&adv_3/StatefulPartitionedCall:output:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
tf.math.reduce_mean/Mean?
tf.math.subtract/SubSubtf.__operators__.add/AddV2:z:0!tf.math.reduce_mean/Mean:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract/Sub?
IdentityIdentitytf.math.subtract/Sub:z:0^adv_1/StatefulPartitionedCall^adv_2/StatefulPartitionedCall^adv_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^val_1/StatefulPartitionedCall^val_2/StatefulPartitionedCall^val_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:??????????: : : : : : : : : : : : : : 2>
adv_1/StatefulPartitionedCalladv_1/StatefulPartitionedCall2>
adv_2/StatefulPartitionedCalladv_2/StatefulPartitionedCall2>
adv_3/StatefulPartitionedCalladv_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
val_1/StatefulPartitionedCallval_1/StatefulPartitionedCall2>
val_2/StatefulPartitionedCallval_2/StatefulPartitionedCall2>
val_3/StatefulPartitionedCallval_3/StatefulPartitionedCall:T P
'
_output_shapes
:??????????
%
_user_specified_nameobservation
??
?
$__inference__traced_restore_28345597
file_prefix2
assignvariableop_dense_1_kernel:	??.
assignvariableop_1_dense_1_bias:	?2
assignvariableop_2_val_1_kernel:	?@+
assignvariableop_3_val_1_bias:@2
assignvariableop_4_adv_1_kernel:	?@+
assignvariableop_5_adv_1_bias:@1
assignvariableop_6_val_2_kernel:@@+
assignvariableop_7_val_2_bias:@1
assignvariableop_8_adv_2_kernel:@@+
assignvariableop_9_adv_2_bias:@2
 assignvariableop_10_val_3_kernel:@,
assignvariableop_11_val_3_bias:2
 assignvariableop_12_adv_3_kernel:@,
assignvariableop_13_adv_3_bias:'
assignvariableop_14_adam_iter:	 )
assignvariableop_15_adam_beta_1: )
assignvariableop_16_adam_beta_2: (
assignvariableop_17_adam_decay: 0
&assignvariableop_18_adam_learning_rate: #
assignvariableop_19_total: #
assignvariableop_20_count: <
)assignvariableop_21_adam_dense_1_kernel_m:	??6
'assignvariableop_22_adam_dense_1_bias_m:	?:
'assignvariableop_23_adam_val_1_kernel_m:	?@3
%assignvariableop_24_adam_val_1_bias_m:@:
'assignvariableop_25_adam_adv_1_kernel_m:	?@3
%assignvariableop_26_adam_adv_1_bias_m:@9
'assignvariableop_27_adam_val_2_kernel_m:@@3
%assignvariableop_28_adam_val_2_bias_m:@9
'assignvariableop_29_adam_adv_2_kernel_m:@@3
%assignvariableop_30_adam_adv_2_bias_m:@9
'assignvariableop_31_adam_val_3_kernel_m:@3
%assignvariableop_32_adam_val_3_bias_m:9
'assignvariableop_33_adam_adv_3_kernel_m:@3
%assignvariableop_34_adam_adv_3_bias_m:<
)assignvariableop_35_adam_dense_1_kernel_v:	??6
'assignvariableop_36_adam_dense_1_bias_v:	?:
'assignvariableop_37_adam_val_1_kernel_v:	?@3
%assignvariableop_38_adam_val_1_bias_v:@:
'assignvariableop_39_adam_adv_1_kernel_v:	?@3
%assignvariableop_40_adam_adv_1_bias_v:@9
'assignvariableop_41_adam_val_2_kernel_v:@@3
%assignvariableop_42_adam_val_2_bias_v:@9
'assignvariableop_43_adam_adv_2_kernel_v:@@3
%assignvariableop_44_adam_adv_2_bias_v:@9
'assignvariableop_45_adam_val_3_kernel_v:@3
%assignvariableop_46_adam_val_3_bias_v:9
'assignvariableop_47_adam_adv_3_kernel_v:@3
%assignvariableop_48_adam_adv_3_bias_v:
identity_50??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*?
value?B?2B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_val_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_val_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adv_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adv_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_val_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_val_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adv_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adv_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp assignvariableop_10_val_3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_val_3_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp assignvariableop_12_adv_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adv_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_val_1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_val_1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_adv_1_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp%assignvariableop_26_adam_adv_1_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_val_2_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_val_2_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_adv_2_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp%assignvariableop_30_adam_adv_2_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_val_3_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_val_3_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp'assignvariableop_33_adam_adv_3_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp%assignvariableop_34_adam_adv_3_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_1_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_1_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_val_1_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp%assignvariableop_38_adam_val_1_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp'assignvariableop_39_adam_adv_1_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp%assignvariableop_40_adam_adv_1_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp'assignvariableop_41_adam_val_2_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp%assignvariableop_42_adam_val_2_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp'assignvariableop_43_adam_adv_2_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp%assignvariableop_44_adam_adv_2_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp'assignvariableop_45_adam_val_3_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp%assignvariableop_46_adam_val_3_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp'assignvariableop_47_adam_adv_3_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp%assignvariableop_48_adam_adv_3_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_489
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?	
Identity_49Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_49?	
Identity_50IdentityIdentity_49:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_50"#
identity_50Identity_50:output:0*w
_input_shapesf
d: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
(__inference_model_layer_call_fn_28345099

inputs
unknown:	??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:	?@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:

unknown_11:@

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_283445862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:??????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
C__inference_adv_3_layer_call_and_return_conditional_losses_28344559

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
C__inference_val_2_layer_call_and_return_conditional_losses_28344543

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
C__inference_adv_1_layer_call_and_return_conditional_losses_28344492

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_adv_1_layer_call_and_return_conditional_losses_28345183

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?+
?
C__inference_model_layer_call_and_return_conditional_losses_28344586

inputs#
dense_1_28344476:	??
dense_1_28344478:	?!
adv_1_28344493:	?@
adv_1_28344495:@!
val_1_28344510:	?@
val_1_28344512:@ 
adv_2_28344527:@@
adv_2_28344529:@ 
val_2_28344544:@@
val_2_28344546:@ 
adv_3_28344560:@
adv_3_28344562: 
val_3_28344576:@
val_3_28344578:
identity??adv_1/StatefulPartitionedCall?adv_2/StatefulPartitionedCall?adv_3/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?val_1/StatefulPartitionedCall?val_2/StatefulPartitionedCall?val_3/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_28344476dense_1_28344478*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_283444752!
dense_1/StatefulPartitionedCall?
adv_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0adv_1_28344493adv_1_28344495*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_adv_1_layer_call_and_return_conditional_losses_283444922
adv_1/StatefulPartitionedCall?
val_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0val_1_28344510val_1_28344512*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_val_1_layer_call_and_return_conditional_losses_283445092
val_1/StatefulPartitionedCall?
adv_2/StatefulPartitionedCallStatefulPartitionedCall&adv_1/StatefulPartitionedCall:output:0adv_2_28344527adv_2_28344529*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_adv_2_layer_call_and_return_conditional_losses_283445262
adv_2/StatefulPartitionedCall?
val_2/StatefulPartitionedCallStatefulPartitionedCall&val_1/StatefulPartitionedCall:output:0val_2_28344544val_2_28344546*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_val_2_layer_call_and_return_conditional_losses_283445432
val_2/StatefulPartitionedCall?
adv_3/StatefulPartitionedCallStatefulPartitionedCall&adv_2/StatefulPartitionedCall:output:0adv_3_28344560adv_3_28344562*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_adv_3_layer_call_and_return_conditional_losses_283445592
adv_3/StatefulPartitionedCall?
val_3/StatefulPartitionedCallStatefulPartitionedCall&val_2/StatefulPartitionedCall:output:0val_3_28344576val_3_28344578*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_val_3_layer_call_and_return_conditional_losses_283445752
val_3/StatefulPartitionedCall?
tf.__operators__.add/AddV2AddV2&val_3/StatefulPartitionedCall:output:0&adv_3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add/AddV2?
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_mean/Mean/reduction_indices?
tf.math.reduce_mean/MeanMean&adv_3/StatefulPartitionedCall:output:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
tf.math.reduce_mean/Mean?
tf.math.subtract/SubSubtf.__operators__.add/AddV2:z:0!tf.math.reduce_mean/Mean:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract/Sub?
IdentityIdentitytf.math.subtract/Sub:z:0^adv_1/StatefulPartitionedCall^adv_2/StatefulPartitionedCall^adv_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^val_1/StatefulPartitionedCall^val_2/StatefulPartitionedCall^val_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:??????????: : : : : : : : : : : : : : 2>
adv_1/StatefulPartitionedCalladv_1/StatefulPartitionedCall2>
adv_2/StatefulPartitionedCalladv_2/StatefulPartitionedCall2>
adv_3/StatefulPartitionedCalladv_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
val_1/StatefulPartitionedCallval_1/StatefulPartitionedCall2>
val_2/StatefulPartitionedCallval_2/StatefulPartitionedCall2>
val_3/StatefulPartitionedCallval_3/StatefulPartitionedCall:O K
'
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_adv_1_layer_call_fn_28345192

inputs
unknown:	?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_adv_1_layer_call_and_return_conditional_losses_283444922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_val_1_layer_call_fn_28345172

inputs
unknown:	?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_val_1_layer_call_and_return_conditional_losses_283445092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_model_layer_call_fn_28344617
observation
unknown:	??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:	?@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:

unknown_11:@

unknown_12:
identity??StatefulPartitionedCall?
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
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_283445862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:??????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:??????????
%
_user_specified_nameobservation
?
?
(__inference_model_layer_call_fn_28345132

inputs
unknown:	??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:	?@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:

unknown_11:@

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_283447652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:??????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
C__inference_val_3_layer_call_and_return_conditional_losses_28345242

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?C
?

C__inference_model_layer_call_and_return_conditional_losses_28345066

inputs9
&dense_1_matmul_readvariableop_resource:	??6
'dense_1_biasadd_readvariableop_resource:	?7
$adv_1_matmul_readvariableop_resource:	?@3
%adv_1_biasadd_readvariableop_resource:@7
$val_1_matmul_readvariableop_resource:	?@3
%val_1_biasadd_readvariableop_resource:@6
$adv_2_matmul_readvariableop_resource:@@3
%adv_2_biasadd_readvariableop_resource:@6
$val_2_matmul_readvariableop_resource:@@3
%val_2_biasadd_readvariableop_resource:@6
$adv_3_matmul_readvariableop_resource:@3
%adv_3_biasadd_readvariableop_resource:6
$val_3_matmul_readvariableop_resource:@3
%val_3_biasadd_readvariableop_resource:
identity??adv_1/BiasAdd/ReadVariableOp?adv_1/MatMul/ReadVariableOp?adv_2/BiasAdd/ReadVariableOp?adv_2/MatMul/ReadVariableOp?adv_3/BiasAdd/ReadVariableOp?adv_3/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?val_1/BiasAdd/ReadVariableOp?val_1/MatMul/ReadVariableOp?val_2/BiasAdd/ReadVariableOp?val_2/MatMul/ReadVariableOp?val_3/BiasAdd/ReadVariableOp?val_3/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Relu?
adv_1/MatMul/ReadVariableOpReadVariableOp$adv_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
adv_1/MatMul/ReadVariableOp?
adv_1/MatMulMatMuldense_1/Relu:activations:0#adv_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
adv_1/MatMul?
adv_1/BiasAdd/ReadVariableOpReadVariableOp%adv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
adv_1/BiasAdd/ReadVariableOp?
adv_1/BiasAddBiasAddadv_1/MatMul:product:0$adv_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
adv_1/BiasAddj

adv_1/ReluReluadv_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2

adv_1/Relu?
val_1/MatMul/ReadVariableOpReadVariableOp$val_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
val_1/MatMul/ReadVariableOp?
val_1/MatMulMatMuldense_1/Relu:activations:0#val_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
val_1/MatMul?
val_1/BiasAdd/ReadVariableOpReadVariableOp%val_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
val_1/BiasAdd/ReadVariableOp?
val_1/BiasAddBiasAddval_1/MatMul:product:0$val_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
val_1/BiasAddj

val_1/ReluReluval_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2

val_1/Relu?
adv_2/MatMul/ReadVariableOpReadVariableOp$adv_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
adv_2/MatMul/ReadVariableOp?
adv_2/MatMulMatMuladv_1/Relu:activations:0#adv_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
adv_2/MatMul?
adv_2/BiasAdd/ReadVariableOpReadVariableOp%adv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
adv_2/BiasAdd/ReadVariableOp?
adv_2/BiasAddBiasAddadv_2/MatMul:product:0$adv_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
adv_2/BiasAddj

adv_2/ReluReluadv_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2

adv_2/Relu?
val_2/MatMul/ReadVariableOpReadVariableOp$val_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
val_2/MatMul/ReadVariableOp?
val_2/MatMulMatMulval_1/Relu:activations:0#val_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
val_2/MatMul?
val_2/BiasAdd/ReadVariableOpReadVariableOp%val_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
val_2/BiasAdd/ReadVariableOp?
val_2/BiasAddBiasAddval_2/MatMul:product:0$val_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
val_2/BiasAddj

val_2/ReluReluval_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2

val_2/Relu?
adv_3/MatMul/ReadVariableOpReadVariableOp$adv_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
adv_3/MatMul/ReadVariableOp?
adv_3/MatMulMatMuladv_2/Relu:activations:0#adv_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
adv_3/MatMul?
adv_3/BiasAdd/ReadVariableOpReadVariableOp%adv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
adv_3/BiasAdd/ReadVariableOp?
adv_3/BiasAddBiasAddadv_3/MatMul:product:0$adv_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
adv_3/BiasAdd?
val_3/MatMul/ReadVariableOpReadVariableOp$val_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
val_3/MatMul/ReadVariableOp?
val_3/MatMulMatMulval_2/Relu:activations:0#val_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
val_3/MatMul?
val_3/BiasAdd/ReadVariableOpReadVariableOp%val_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
val_3/BiasAdd/ReadVariableOp?
val_3/BiasAddBiasAddval_3/MatMul:product:0$val_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
val_3/BiasAdd?
tf.__operators__.add/AddV2AddV2val_3/BiasAdd:output:0adv_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add/AddV2?
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_mean/Mean/reduction_indices?
tf.math.reduce_mean/MeanMeanadv_3/BiasAdd:output:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
tf.math.reduce_mean/Mean?
tf.math.subtract/SubSubtf.__operators__.add/AddV2:z:0!tf.math.reduce_mean/Mean:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract/Sub?
IdentityIdentitytf.math.subtract/Sub:z:0^adv_1/BiasAdd/ReadVariableOp^adv_1/MatMul/ReadVariableOp^adv_2/BiasAdd/ReadVariableOp^adv_2/MatMul/ReadVariableOp^adv_3/BiasAdd/ReadVariableOp^adv_3/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^val_1/BiasAdd/ReadVariableOp^val_1/MatMul/ReadVariableOp^val_2/BiasAdd/ReadVariableOp^val_2/MatMul/ReadVariableOp^val_3/BiasAdd/ReadVariableOp^val_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:??????????: : : : : : : : : : : : : : 2<
adv_1/BiasAdd/ReadVariableOpadv_1/BiasAdd/ReadVariableOp2:
adv_1/MatMul/ReadVariableOpadv_1/MatMul/ReadVariableOp2<
adv_2/BiasAdd/ReadVariableOpadv_2/BiasAdd/ReadVariableOp2:
adv_2/MatMul/ReadVariableOpadv_2/MatMul/ReadVariableOp2<
adv_3/BiasAdd/ReadVariableOpadv_3/BiasAdd/ReadVariableOp2:
adv_3/MatMul/ReadVariableOpadv_3/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2<
val_1/BiasAdd/ReadVariableOpval_1/BiasAdd/ReadVariableOp2:
val_1/MatMul/ReadVariableOpval_1/MatMul/ReadVariableOp2<
val_2/BiasAdd/ReadVariableOpval_2/BiasAdd/ReadVariableOp2:
val_2/MatMul/ReadVariableOpval_2/MatMul/ReadVariableOp2<
val_3/BiasAdd/ReadVariableOpval_3/BiasAdd/ReadVariableOp2:
val_3/MatMul/ReadVariableOpval_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_val_3_layer_call_fn_28345251

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_val_3_layer_call_and_return_conditional_losses_283445752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?b
?
!__inference__traced_save_28345440
file_prefix-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop+
'savev2_val_1_kernel_read_readvariableop)
%savev2_val_1_bias_read_readvariableop+
'savev2_adv_1_kernel_read_readvariableop)
%savev2_adv_1_bias_read_readvariableop+
'savev2_val_2_kernel_read_readvariableop)
%savev2_val_2_bias_read_readvariableop+
'savev2_adv_2_kernel_read_readvariableop)
%savev2_adv_2_bias_read_readvariableop+
'savev2_val_3_kernel_read_readvariableop)
%savev2_val_3_bias_read_readvariableop+
'savev2_adv_3_kernel_read_readvariableop)
%savev2_adv_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop2
.savev2_adam_val_1_kernel_m_read_readvariableop0
,savev2_adam_val_1_bias_m_read_readvariableop2
.savev2_adam_adv_1_kernel_m_read_readvariableop0
,savev2_adam_adv_1_bias_m_read_readvariableop2
.savev2_adam_val_2_kernel_m_read_readvariableop0
,savev2_adam_val_2_bias_m_read_readvariableop2
.savev2_adam_adv_2_kernel_m_read_readvariableop0
,savev2_adam_adv_2_bias_m_read_readvariableop2
.savev2_adam_val_3_kernel_m_read_readvariableop0
,savev2_adam_val_3_bias_m_read_readvariableop2
.savev2_adam_adv_3_kernel_m_read_readvariableop0
,savev2_adam_adv_3_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop2
.savev2_adam_val_1_kernel_v_read_readvariableop0
,savev2_adam_val_1_bias_v_read_readvariableop2
.savev2_adam_adv_1_kernel_v_read_readvariableop0
,savev2_adam_adv_1_bias_v_read_readvariableop2
.savev2_adam_val_2_kernel_v_read_readvariableop0
,savev2_adam_val_2_bias_v_read_readvariableop2
.savev2_adam_adv_2_kernel_v_read_readvariableop0
,savev2_adam_adv_2_bias_v_read_readvariableop2
.savev2_adam_val_3_kernel_v_read_readvariableop0
,savev2_adam_val_3_bias_v_read_readvariableop2
.savev2_adam_adv_3_kernel_v_read_readvariableop0
,savev2_adam_adv_3_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*?
value?B?2B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop'savev2_val_1_kernel_read_readvariableop%savev2_val_1_bias_read_readvariableop'savev2_adv_1_kernel_read_readvariableop%savev2_adv_1_bias_read_readvariableop'savev2_val_2_kernel_read_readvariableop%savev2_val_2_bias_read_readvariableop'savev2_adv_2_kernel_read_readvariableop%savev2_adv_2_bias_read_readvariableop'savev2_val_3_kernel_read_readvariableop%savev2_val_3_bias_read_readvariableop'savev2_adv_3_kernel_read_readvariableop%savev2_adv_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop.savev2_adam_val_1_kernel_m_read_readvariableop,savev2_adam_val_1_bias_m_read_readvariableop.savev2_adam_adv_1_kernel_m_read_readvariableop,savev2_adam_adv_1_bias_m_read_readvariableop.savev2_adam_val_2_kernel_m_read_readvariableop,savev2_adam_val_2_bias_m_read_readvariableop.savev2_adam_adv_2_kernel_m_read_readvariableop,savev2_adam_adv_2_bias_m_read_readvariableop.savev2_adam_val_3_kernel_m_read_readvariableop,savev2_adam_val_3_bias_m_read_readvariableop.savev2_adam_adv_3_kernel_m_read_readvariableop,savev2_adam_adv_3_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop.savev2_adam_val_1_kernel_v_read_readvariableop,savev2_adam_val_1_bias_v_read_readvariableop.savev2_adam_adv_1_kernel_v_read_readvariableop,savev2_adam_adv_1_bias_v_read_readvariableop.savev2_adam_val_2_kernel_v_read_readvariableop,savev2_adam_val_2_bias_v_read_readvariableop.savev2_adam_adv_2_kernel_v_read_readvariableop,savev2_adam_adv_2_bias_v_read_readvariableop.savev2_adam_val_3_kernel_v_read_readvariableop,savev2_adam_val_3_bias_v_read_readvariableop.savev2_adam_adv_3_kernel_v_read_readvariableop,savev2_adam_adv_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *@
dtypes6
422	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	??:?:	?@:@:	?@:@:@@:@:@@:@:@::@:: : : : : : : :	??:?:	?@:@:	?@:@:@@:@:@@:@:@::@::	??:?:	?@:@:	?@:@:@@:@:@@:@:@::@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	??:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$	 

_output_shapes

:@@: 


_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	??:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$  

_output_shapes

:@: !

_output_shapes
::$" 

_output_shapes

:@: #

_output_shapes
::%$!

_output_shapes
:	??:!%

_output_shapes	
:?:%&!

_output_shapes
:	?@: '

_output_shapes
:@:%(!

_output_shapes
:	?@: )

_output_shapes
:@:$* 

_output_shapes

:@@: +

_output_shapes
:@:$, 

_output_shapes

:@@: -

_output_shapes
:@:$. 

_output_shapes

:@: /

_output_shapes
::$0 

_output_shapes

:@: 1

_output_shapes
::2

_output_shapes
: 
?

?
C__inference_val_2_layer_call_and_return_conditional_losses_28345203

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
C__inference_val_1_layer_call_and_return_conditional_losses_28344509

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_dense_1_layer_call_and_return_conditional_losses_28344475

inputs1
matmul_readvariableop_resource:	??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_dense_1_layer_call_fn_28345152

inputs
unknown:	??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_283444752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
C__inference_adv_3_layer_call_and_return_conditional_losses_28345261

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
C__inference_val_1_layer_call_and_return_conditional_losses_28345163

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
C__inference_val_3_layer_call_and_return_conditional_losses_28344575

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_model_layer_call_fn_28344829
observation
unknown:	??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:	?@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:

unknown_11:@

unknown_12:
identity??StatefulPartitionedCall?
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
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_283447652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:??????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:??????????
%
_user_specified_nameobservation
?C
?

C__inference_model_layer_call_and_return_conditional_losses_28345011

inputs9
&dense_1_matmul_readvariableop_resource:	??6
'dense_1_biasadd_readvariableop_resource:	?7
$adv_1_matmul_readvariableop_resource:	?@3
%adv_1_biasadd_readvariableop_resource:@7
$val_1_matmul_readvariableop_resource:	?@3
%val_1_biasadd_readvariableop_resource:@6
$adv_2_matmul_readvariableop_resource:@@3
%adv_2_biasadd_readvariableop_resource:@6
$val_2_matmul_readvariableop_resource:@@3
%val_2_biasadd_readvariableop_resource:@6
$adv_3_matmul_readvariableop_resource:@3
%adv_3_biasadd_readvariableop_resource:6
$val_3_matmul_readvariableop_resource:@3
%val_3_biasadd_readvariableop_resource:
identity??adv_1/BiasAdd/ReadVariableOp?adv_1/MatMul/ReadVariableOp?adv_2/BiasAdd/ReadVariableOp?adv_2/MatMul/ReadVariableOp?adv_3/BiasAdd/ReadVariableOp?adv_3/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?val_1/BiasAdd/ReadVariableOp?val_1/MatMul/ReadVariableOp?val_2/BiasAdd/ReadVariableOp?val_2/MatMul/ReadVariableOp?val_3/BiasAdd/ReadVariableOp?val_3/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Relu?
adv_1/MatMul/ReadVariableOpReadVariableOp$adv_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
adv_1/MatMul/ReadVariableOp?
adv_1/MatMulMatMuldense_1/Relu:activations:0#adv_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
adv_1/MatMul?
adv_1/BiasAdd/ReadVariableOpReadVariableOp%adv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
adv_1/BiasAdd/ReadVariableOp?
adv_1/BiasAddBiasAddadv_1/MatMul:product:0$adv_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
adv_1/BiasAddj

adv_1/ReluReluadv_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2

adv_1/Relu?
val_1/MatMul/ReadVariableOpReadVariableOp$val_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
val_1/MatMul/ReadVariableOp?
val_1/MatMulMatMuldense_1/Relu:activations:0#val_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
val_1/MatMul?
val_1/BiasAdd/ReadVariableOpReadVariableOp%val_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
val_1/BiasAdd/ReadVariableOp?
val_1/BiasAddBiasAddval_1/MatMul:product:0$val_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
val_1/BiasAddj

val_1/ReluReluval_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2

val_1/Relu?
adv_2/MatMul/ReadVariableOpReadVariableOp$adv_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
adv_2/MatMul/ReadVariableOp?
adv_2/MatMulMatMuladv_1/Relu:activations:0#adv_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
adv_2/MatMul?
adv_2/BiasAdd/ReadVariableOpReadVariableOp%adv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
adv_2/BiasAdd/ReadVariableOp?
adv_2/BiasAddBiasAddadv_2/MatMul:product:0$adv_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
adv_2/BiasAddj

adv_2/ReluReluadv_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2

adv_2/Relu?
val_2/MatMul/ReadVariableOpReadVariableOp$val_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
val_2/MatMul/ReadVariableOp?
val_2/MatMulMatMulval_1/Relu:activations:0#val_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
val_2/MatMul?
val_2/BiasAdd/ReadVariableOpReadVariableOp%val_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
val_2/BiasAdd/ReadVariableOp?
val_2/BiasAddBiasAddval_2/MatMul:product:0$val_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
val_2/BiasAddj

val_2/ReluReluval_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2

val_2/Relu?
adv_3/MatMul/ReadVariableOpReadVariableOp$adv_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
adv_3/MatMul/ReadVariableOp?
adv_3/MatMulMatMuladv_2/Relu:activations:0#adv_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
adv_3/MatMul?
adv_3/BiasAdd/ReadVariableOpReadVariableOp%adv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
adv_3/BiasAdd/ReadVariableOp?
adv_3/BiasAddBiasAddadv_3/MatMul:product:0$adv_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
adv_3/BiasAdd?
val_3/MatMul/ReadVariableOpReadVariableOp$val_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
val_3/MatMul/ReadVariableOp?
val_3/MatMulMatMulval_2/Relu:activations:0#val_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
val_3/MatMul?
val_3/BiasAdd/ReadVariableOpReadVariableOp%val_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
val_3/BiasAdd/ReadVariableOp?
val_3/BiasAddBiasAddval_3/MatMul:product:0$val_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
val_3/BiasAdd?
tf.__operators__.add/AddV2AddV2val_3/BiasAdd:output:0adv_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add/AddV2?
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_mean/Mean/reduction_indices?
tf.math.reduce_mean/MeanMeanadv_3/BiasAdd:output:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
tf.math.reduce_mean/Mean?
tf.math.subtract/SubSubtf.__operators__.add/AddV2:z:0!tf.math.reduce_mean/Mean:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract/Sub?
IdentityIdentitytf.math.subtract/Sub:z:0^adv_1/BiasAdd/ReadVariableOp^adv_1/MatMul/ReadVariableOp^adv_2/BiasAdd/ReadVariableOp^adv_2/MatMul/ReadVariableOp^adv_3/BiasAdd/ReadVariableOp^adv_3/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^val_1/BiasAdd/ReadVariableOp^val_1/MatMul/ReadVariableOp^val_2/BiasAdd/ReadVariableOp^val_2/MatMul/ReadVariableOp^val_3/BiasAdd/ReadVariableOp^val_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:??????????: : : : : : : : : : : : : : 2<
adv_1/BiasAdd/ReadVariableOpadv_1/BiasAdd/ReadVariableOp2:
adv_1/MatMul/ReadVariableOpadv_1/MatMul/ReadVariableOp2<
adv_2/BiasAdd/ReadVariableOpadv_2/BiasAdd/ReadVariableOp2:
adv_2/MatMul/ReadVariableOpadv_2/MatMul/ReadVariableOp2<
adv_3/BiasAdd/ReadVariableOpadv_3/BiasAdd/ReadVariableOp2:
adv_3/MatMul/ReadVariableOpadv_3/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2<
val_1/BiasAdd/ReadVariableOpval_1/BiasAdd/ReadVariableOp2:
val_1/MatMul/ReadVariableOpval_1/MatMul/ReadVariableOp2<
val_2/BiasAdd/ReadVariableOpval_2/BiasAdd/ReadVariableOp2:
val_2/MatMul/ReadVariableOpval_2/MatMul/ReadVariableOp2<
val_3/BiasAdd/ReadVariableOpval_3/BiasAdd/ReadVariableOp2:
val_3/MatMul/ReadVariableOpval_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:??????????
 
_user_specified_nameinputs
?+
?
C__inference_model_layer_call_and_return_conditional_losses_28344872
observation#
dense_1_28344832:	??
dense_1_28344834:	?!
adv_1_28344837:	?@
adv_1_28344839:@!
val_1_28344842:	?@
val_1_28344844:@ 
adv_2_28344847:@@
adv_2_28344849:@ 
val_2_28344852:@@
val_2_28344854:@ 
adv_3_28344857:@
adv_3_28344859: 
val_3_28344862:@
val_3_28344864:
identity??adv_1/StatefulPartitionedCall?adv_2/StatefulPartitionedCall?adv_3/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?val_1/StatefulPartitionedCall?val_2/StatefulPartitionedCall?val_3/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallobservationdense_1_28344832dense_1_28344834*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_283444752!
dense_1/StatefulPartitionedCall?
adv_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0adv_1_28344837adv_1_28344839*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_adv_1_layer_call_and_return_conditional_losses_283444922
adv_1/StatefulPartitionedCall?
val_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0val_1_28344842val_1_28344844*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_val_1_layer_call_and_return_conditional_losses_283445092
val_1/StatefulPartitionedCall?
adv_2/StatefulPartitionedCallStatefulPartitionedCall&adv_1/StatefulPartitionedCall:output:0adv_2_28344847adv_2_28344849*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_adv_2_layer_call_and_return_conditional_losses_283445262
adv_2/StatefulPartitionedCall?
val_2/StatefulPartitionedCallStatefulPartitionedCall&val_1/StatefulPartitionedCall:output:0val_2_28344852val_2_28344854*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_val_2_layer_call_and_return_conditional_losses_283445432
val_2/StatefulPartitionedCall?
adv_3/StatefulPartitionedCallStatefulPartitionedCall&adv_2/StatefulPartitionedCall:output:0adv_3_28344857adv_3_28344859*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_adv_3_layer_call_and_return_conditional_losses_283445592
adv_3/StatefulPartitionedCall?
val_3/StatefulPartitionedCallStatefulPartitionedCall&val_2/StatefulPartitionedCall:output:0val_3_28344862val_3_28344864*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_val_3_layer_call_and_return_conditional_losses_283445752
val_3/StatefulPartitionedCall?
tf.__operators__.add/AddV2AddV2&val_3/StatefulPartitionedCall:output:0&adv_3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add/AddV2?
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_mean/Mean/reduction_indices?
tf.math.reduce_mean/MeanMean&adv_3/StatefulPartitionedCall:output:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
tf.math.reduce_mean/Mean?
tf.math.subtract/SubSubtf.__operators__.add/AddV2:z:0!tf.math.reduce_mean/Mean:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract/Sub?
IdentityIdentitytf.math.subtract/Sub:z:0^adv_1/StatefulPartitionedCall^adv_2/StatefulPartitionedCall^adv_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^val_1/StatefulPartitionedCall^val_2/StatefulPartitionedCall^val_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:??????????: : : : : : : : : : : : : : 2>
adv_1/StatefulPartitionedCalladv_1/StatefulPartitionedCall2>
adv_2/StatefulPartitionedCalladv_2/StatefulPartitionedCall2>
adv_3/StatefulPartitionedCalladv_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
val_1/StatefulPartitionedCallval_1/StatefulPartitionedCall2>
val_2/StatefulPartitionedCallval_2/StatefulPartitionedCall2>
val_3/StatefulPartitionedCallval_3/StatefulPartitionedCall:T P
'
_output_shapes
:??????????
%
_user_specified_nameobservation
?

?
C__inference_adv_2_layer_call_and_return_conditional_losses_28344526

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_adv_2_layer_call_fn_28345232

inputs
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_adv_2_layer_call_and_return_conditional_losses_283445262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
observation4
serving_default_observation:0??????????D
tf.math.subtract0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:Ġ
?X
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
	variables
regularization_losses
trainable_variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"?T
_tf_keras_network?T{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 63]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "observation"}, "name": "observation", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["observation", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "val_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "val_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "adv_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "adv_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "val_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "val_2", "inbound_nodes": [[["val_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "adv_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "adv_2", "inbound_nodes": [[["adv_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "val_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "val_3", "inbound_nodes": [[["val_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "adv_3", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "adv_3", "inbound_nodes": [[["adv_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add", "inbound_nodes": [["val_3", 0, 0, {"y": ["adv_3", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean", "inbound_nodes": [["adv_3", 0, 0, {"axis": 1, "keepdims": true}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract", "inbound_nodes": [["tf.__operators__.add", 0, 0, {"y": ["tf.math.reduce_mean", 0, 0], "name": null}]]}], "input_layers": [["observation", 0, 0]], "output_layers": [["tf.math.subtract", 0, 0]]}, "shared_object_id": 25, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 63]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 63]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 63]}, "float32", "observation"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 63]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "observation"}, "name": "observation", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["observation", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Dense", "config": {"name": "val_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "val_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "adv_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "adv_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dense", "config": {"name": "val_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "val_2", "inbound_nodes": [[["val_1", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Dense", "config": {"name": "adv_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "adv_2", "inbound_nodes": [[["adv_1", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "Dense", "config": {"name": "val_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "val_3", "inbound_nodes": [[["val_2", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "Dense", "config": {"name": "adv_3", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "adv_3", "inbound_nodes": [[["adv_2", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add", "inbound_nodes": [["val_3", 0, 0, {"y": ["adv_3", 0, 0], "name": null}]], "shared_object_id": 22}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean", "inbound_nodes": [["adv_3", 0, 0, {"axis": 1, "keepdims": true}]], "shared_object_id": 23}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract", "inbound_nodes": [["tf.__operators__.add", 0, 0, {"y": ["tf.math.reduce_mean", 0, 0], "name": null}]], "shared_object_id": 24}], "input_layers": [["observation", 0, 0]], "output_layers": [["tf.math.subtract", 0, 0]]}}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}, "shared_object_id": 27}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "observation", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 63]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 63]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "observation"}}
?	

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["observation", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 63}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 63]}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "val_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "val_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 29}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "adv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "adv_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 30}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

$kernel
%bias
&	variables
'regularization_losses
(trainable_variables
)	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "val_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "val_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["val_1", 0, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

*kernel
+bias
,	variables
-regularization_losses
.trainable_variables
/	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "adv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "adv_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["adv_1", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 32}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

0kernel
1bias
2	variables
3regularization_losses
4trainable_variables
5	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "val_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "val_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["val_2", 0, 0, {}]]], "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

6kernel
7bias
8	variables
9regularization_losses
:trainable_variables
;	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "adv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "adv_3", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["adv_2", 0, 0, {}]]], "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 34}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
<	keras_api"?
_tf_keras_layer?{"name": "tf.__operators__.add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "inbound_nodes": [["val_3", 0, 0, {"y": ["adv_3", 0, 0], "name": null}]], "shared_object_id": 22}
?
=	keras_api"?
_tf_keras_layer?{"name": "tf.math.reduce_mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "inbound_nodes": [["adv_3", 0, 0, {"axis": 1, "keepdims": true}]], "shared_object_id": 23}
?
>	keras_api"?
_tf_keras_layer?{"name": "tf.math.subtract", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.subtract", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "inbound_nodes": [["tf.__operators__.add", 0, 0, {"y": ["tf.math.reduce_mean", 0, 0], "name": null}]], "shared_object_id": 24}
?
?iter

@beta_1

Abeta_2
	Bdecay
Clearning_ratemqmrmsmtmumv$mw%mx*my+mz0m{1m|6m}7m~vv?v?v?v?v?$v?%v?*v?+v?0v?1v?6v?7v?"
	optimizer
?
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
?
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
?
Dnon_trainable_variables

Elayers
Flayer_metrics
Glayer_regularization_losses
	variables
regularization_losses
trainable_variables
Hmetrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
!:	??2dense_1/kernel
:?2dense_1/bias
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
?
Inon_trainable_variables

Jlayers
Klayer_metrics
Llayer_regularization_losses
	variables
regularization_losses
trainable_variables
Mmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	?@2val_1/kernel
:@2
val_1/bias
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
?
Nnon_trainable_variables

Olayers
Player_metrics
Qlayer_regularization_losses
	variables
regularization_losses
trainable_variables
Rmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	?@2adv_1/kernel
:@2
adv_1/bias
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
?
Snon_trainable_variables

Tlayers
Ulayer_metrics
Vlayer_regularization_losses
 	variables
!regularization_losses
"trainable_variables
Wmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:@@2val_2/kernel
:@2
val_2/bias
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
?
Xnon_trainable_variables

Ylayers
Zlayer_metrics
[layer_regularization_losses
&	variables
'regularization_losses
(trainable_variables
\metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:@@2adv_2/kernel
:@2
adv_2/bias
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
?
]non_trainable_variables

^layers
_layer_metrics
`layer_regularization_losses
,	variables
-regularization_losses
.trainable_variables
ametrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:@2val_3/kernel
:2
val_3/bias
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
?
bnon_trainable_variables

clayers
dlayer_metrics
elayer_regularization_losses
2	variables
3regularization_losses
4trainable_variables
fmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:@2adv_3/kernel
:2
adv_3/bias
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
?
gnon_trainable_variables

hlayers
ilayer_metrics
jlayer_regularization_losses
8	variables
9regularization_losses
:trainable_variables
kmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
'
l0"
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
 "
trackable_list_wrapper
?
	mtotal
	ncount
o	variables
p	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 35}
:  (2total
:  (2count
.
m0
n1"
trackable_list_wrapper
-
o	variables"
_generic_user_object
&:$	??2Adam/dense_1/kernel/m
 :?2Adam/dense_1/bias/m
$:"	?@2Adam/val_1/kernel/m
:@2Adam/val_1/bias/m
$:"	?@2Adam/adv_1/kernel/m
:@2Adam/adv_1/bias/m
#:!@@2Adam/val_2/kernel/m
:@2Adam/val_2/bias/m
#:!@@2Adam/adv_2/kernel/m
:@2Adam/adv_2/bias/m
#:!@2Adam/val_3/kernel/m
:2Adam/val_3/bias/m
#:!@2Adam/adv_3/kernel/m
:2Adam/adv_3/bias/m
&:$	??2Adam/dense_1/kernel/v
 :?2Adam/dense_1/bias/v
$:"	?@2Adam/val_1/kernel/v
:@2Adam/val_1/bias/v
$:"	?@2Adam/adv_1/kernel/v
:@2Adam/adv_1/bias/v
#:!@@2Adam/val_2/kernel/v
:@2Adam/val_2/bias/v
#:!@@2Adam/adv_2/kernel/v
:@2Adam/adv_2/bias/v
#:!@2Adam/val_3/kernel/v
:2Adam/val_3/bias/v
#:!@2Adam/adv_3/kernel/v
:2Adam/adv_3/bias/v
?2?
C__inference_model_layer_call_and_return_conditional_losses_28345011
C__inference_model_layer_call_and_return_conditional_losses_28345066
C__inference_model_layer_call_and_return_conditional_losses_28344872
C__inference_model_layer_call_and_return_conditional_losses_28344915?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_model_layer_call_fn_28344617
(__inference_model_layer_call_fn_28345099
(__inference_model_layer_call_fn_28345132
(__inference_model_layer_call_fn_28344829?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference__wrapped_model_28344457?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"
observation??????????
?2?
E__inference_dense_1_layer_call_and_return_conditional_losses_28345143?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_1_layer_call_fn_28345152?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_val_1_layer_call_and_return_conditional_losses_28345163?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_val_1_layer_call_fn_28345172?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_adv_1_layer_call_and_return_conditional_losses_28345183?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_adv_1_layer_call_fn_28345192?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_val_2_layer_call_and_return_conditional_losses_28345203?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_val_2_layer_call_fn_28345212?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_adv_2_layer_call_and_return_conditional_losses_28345223?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_adv_2_layer_call_fn_28345232?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_val_3_layer_call_and_return_conditional_losses_28345242?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_val_3_layer_call_fn_28345251?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_adv_3_layer_call_and_return_conditional_losses_28345261?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_adv_3_layer_call_fn_28345270?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_28344956observation"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_28344457?*+$%67014?1
*?'
%?"
observation??????????
? "C?@
>
tf.math.subtract*?'
tf.math.subtract??????????
C__inference_adv_1_layer_call_and_return_conditional_losses_28345183]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? |
(__inference_adv_1_layer_call_fn_28345192P0?-
&?#
!?
inputs??????????
? "??????????@?
C__inference_adv_2_layer_call_and_return_conditional_losses_28345223\*+/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? {
(__inference_adv_2_layer_call_fn_28345232O*+/?,
%?"
 ?
inputs?????????@
? "??????????@?
C__inference_adv_3_layer_call_and_return_conditional_losses_28345261\67/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? {
(__inference_adv_3_layer_call_fn_28345270O67/?,
%?"
 ?
inputs?????????@
? "???????????
E__inference_dense_1_layer_call_and_return_conditional_losses_28345143]/?,
%?"
 ?
inputs??????????
? "&?#
?
0??????????
? ~
*__inference_dense_1_layer_call_fn_28345152P/?,
%?"
 ?
inputs??????????
? "????????????
C__inference_model_layer_call_and_return_conditional_losses_28344872u*+$%6701<?9
2?/
%?"
observation??????????
p 

 
? "%?"
?
0?????????
? ?
C__inference_model_layer_call_and_return_conditional_losses_28344915u*+$%6701<?9
2?/
%?"
observation??????????
p

 
? "%?"
?
0?????????
? ?
C__inference_model_layer_call_and_return_conditional_losses_28345011p*+$%67017?4
-?*
 ?
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
C__inference_model_layer_call_and_return_conditional_losses_28345066p*+$%67017?4
-?*
 ?
inputs??????????
p

 
? "%?"
?
0?????????
? ?
(__inference_model_layer_call_fn_28344617h*+$%6701<?9
2?/
%?"
observation??????????
p 

 
? "???????????
(__inference_model_layer_call_fn_28344829h*+$%6701<?9
2?/
%?"
observation??????????
p

 
? "???????????
(__inference_model_layer_call_fn_28345099c*+$%67017?4
-?*
 ?
inputs??????????
p 

 
? "???????????
(__inference_model_layer_call_fn_28345132c*+$%67017?4
-?*
 ?
inputs??????????
p

 
? "???????????
&__inference_signature_wrapper_28344956?*+$%6701C?@
? 
9?6
4
observation%?"
observation??????????"C?@
>
tf.math.subtract*?'
tf.math.subtract??????????
C__inference_val_1_layer_call_and_return_conditional_losses_28345163]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? |
(__inference_val_1_layer_call_fn_28345172P0?-
&?#
!?
inputs??????????
? "??????????@?
C__inference_val_2_layer_call_and_return_conditional_losses_28345203\$%/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? {
(__inference_val_2_layer_call_fn_28345212O$%/?,
%?"
 ?
inputs?????????@
? "??????????@?
C__inference_val_3_layer_call_and_return_conditional_losses_28345242\01/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? {
(__inference_val_3_layer_call_fn_28345251O01/?,
%?"
 ?
inputs?????????@
? "??????????