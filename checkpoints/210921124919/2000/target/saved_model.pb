è
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
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718À
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	?*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:*
dtype0
u
val_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_nameval_1/kernel
n
 val_1/kernel/Read/ReadVariableOpReadVariableOpval_1/kernel*
_output_shapes
:	@*
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
shape:	@*
shared_nameadv_1/kernel
n
 adv_1/kernel/Read/ReadVariableOpReadVariableOpadv_1/kernel*
_output_shapes
:	@*
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

NoOpNoOp
ê%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¥%
value%B% B%
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
?non_trainable_variables

@layers
Alayer_metrics
Blayer_regularization_losses
	variables
regularization_losses
trainable_variables
Cmetrics
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
­
Dnon_trainable_variables

Elayers
Flayer_metrics
Glayer_regularization_losses
	variables
regularization_losses
trainable_variables
Hmetrics
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
­
Inon_trainable_variables

Jlayers
Klayer_metrics
Llayer_regularization_losses
	variables
regularization_losses
trainable_variables
Mmetrics
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
­
Nnon_trainable_variables

Olayers
Player_metrics
Qlayer_regularization_losses
 	variables
!regularization_losses
"trainable_variables
Rmetrics
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
­
Snon_trainable_variables

Tlayers
Ulayer_metrics
Vlayer_regularization_losses
&	variables
'regularization_losses
(trainable_variables
Wmetrics
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
­
Xnon_trainable_variables

Ylayers
Zlayer_metrics
[layer_regularization_losses
,	variables
-regularization_losses
.trainable_variables
\metrics
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
­
]non_trainable_variables

^layers
_layer_metrics
`layer_regularization_losses
2	variables
3regularization_losses
4trainable_variables
ametrics
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
­
bnon_trainable_variables

clayers
dlayer_metrics
elayer_regularization_losses
8	variables
9regularization_losses
:trainable_variables
fmetrics
 
 
 
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
g0
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
	htotal
	icount
j	variables
k	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

h0
i1

j	variables
~
serving_default_observationPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ?

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
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 */
f*R(
&__inference_signature_wrapper_28346369
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¸
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp val_1/kernel/Read/ReadVariableOpval_1/bias/Read/ReadVariableOp adv_1/kernel/Read/ReadVariableOpadv_1/bias/Read/ReadVariableOp val_2/kernel/Read/ReadVariableOpval_2/bias/Read/ReadVariableOp adv_2/kernel/Read/ReadVariableOpadv_2/bias/Read/ReadVariableOp val_3/kernel/Read/ReadVariableOpval_3/bias/Read/ReadVariableOp adv_3/kernel/Read/ReadVariableOpadv_3/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tin
2*
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
GPU2*0J 8 **
f%R#
!__inference__traced_save_28346754
ó
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1/kerneldense_1/biasval_1/kernel
val_1/biasadv_1/kernel
adv_1/biasval_2/kernel
val_2/biasadv_2/kernel
adv_2/biasval_3/kernel
val_3/biasadv_3/kernel
adv_3/biastotalcount*
Tin
2*
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
GPU2*0J 8 *-
f(R&
$__inference__traced_restore_28346812Ïî
Þ+

E__inference_model_1_layer_call_and_return_conditional_losses_28346291
observation#
dense_1_28346251:	?
dense_1_28346253:	!
adv_1_28346256:	@
adv_1_28346258:@!
val_1_28346261:	@
val_1_28346263:@ 
adv_2_28346266:@@
adv_2_28346268:@ 
val_2_28346271:@@
val_2_28346273:@ 
adv_3_28346276:@
adv_3_28346278: 
val_3_28346281:@
val_3_28346283:
identity¢adv_1/StatefulPartitionedCall¢adv_2/StatefulPartitionedCall¢adv_3/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢val_1/StatefulPartitionedCall¢val_2/StatefulPartitionedCall¢val_3/StatefulPartitionedCall
dense_1/StatefulPartitionedCallStatefulPartitionedCallobservationdense_1_28346251dense_1_28346253*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_283458942!
dense_1/StatefulPartitionedCall°
adv_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0adv_1_28346256adv_1_28346258*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_adv_1_layer_call_and_return_conditional_losses_283459112
adv_1/StatefulPartitionedCall°
val_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0val_1_28346261val_1_28346263*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_val_1_layer_call_and_return_conditional_losses_283459282
val_1/StatefulPartitionedCall®
adv_2/StatefulPartitionedCallStatefulPartitionedCall&adv_1/StatefulPartitionedCall:output:0adv_2_28346266adv_2_28346268*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_adv_2_layer_call_and_return_conditional_losses_283459452
adv_2/StatefulPartitionedCall®
val_2/StatefulPartitionedCallStatefulPartitionedCall&val_1/StatefulPartitionedCall:output:0val_2_28346271val_2_28346273*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_val_2_layer_call_and_return_conditional_losses_283459622
val_2/StatefulPartitionedCall®
adv_3/StatefulPartitionedCallStatefulPartitionedCall&adv_2/StatefulPartitionedCall:output:0adv_3_28346276adv_3_28346278*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_adv_3_layer_call_and_return_conditional_losses_283459782
adv_3/StatefulPartitionedCall®
val_3/StatefulPartitionedCallStatefulPartitionedCall&val_2/StatefulPartitionedCall:output:0val_3_28346281val_3_28346283*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_val_3_layer_call_and_return_conditional_losses_283459942
val_3/StatefulPartitionedCallÇ
tf.__operators__.add_1/AddV2AddV2&val_3/StatefulPartitionedCall:output:0&adv_3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_1/AddV2
,tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,tf.math.reduce_mean_1/Mean/reduction_indicesâ
tf.math.reduce_mean_1/MeanMean&adv_3/StatefulPartitionedCall:output:05tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
tf.math.reduce_mean_1/Mean°
tf.math.subtract_1/SubSub tf.__operators__.add_1/AddV2:z:0#tf.math.reduce_mean_1/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.subtract_1/SubÐ
IdentityIdentitytf.math.subtract_1/Sub:z:0^adv_1/StatefulPartitionedCall^adv_2/StatefulPartitionedCall^adv_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^val_1/StatefulPartitionedCall^val_2/StatefulPartitionedCall^val_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ?: : : : : : : : : : : : : : 2>
adv_1/StatefulPartitionedCalladv_1/StatefulPartitionedCall2>
adv_2/StatefulPartitionedCalladv_2/StatefulPartitionedCall2>
adv_3/StatefulPartitionedCalladv_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
val_1/StatefulPartitionedCallval_1/StatefulPartitionedCall2>
val_2/StatefulPartitionedCallval_2/StatefulPartitionedCall2>
val_3/StatefulPartitionedCallval_3/StatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?
%
_user_specified_nameobservation
«

ô
C__inference_adv_2_layer_call_and_return_conditional_losses_28345945

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
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
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


(__inference_val_2_layer_call_fn_28346625

inputs
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_val_2_layer_call_and_return_conditional_losses_283459622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


(__inference_val_3_layer_call_fn_28346664

inputs
unknown:@
	unknown_0:
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_val_3_layer_call_and_return_conditional_losses_283459942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ï+

E__inference_model_1_layer_call_and_return_conditional_losses_28346184

inputs#
dense_1_28346144:	?
dense_1_28346146:	!
adv_1_28346149:	@
adv_1_28346151:@!
val_1_28346154:	@
val_1_28346156:@ 
adv_2_28346159:@@
adv_2_28346161:@ 
val_2_28346164:@@
val_2_28346166:@ 
adv_3_28346169:@
adv_3_28346171: 
val_3_28346174:@
val_3_28346176:
identity¢adv_1/StatefulPartitionedCall¢adv_2/StatefulPartitionedCall¢adv_3/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢val_1/StatefulPartitionedCall¢val_2/StatefulPartitionedCall¢val_3/StatefulPartitionedCall
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_28346144dense_1_28346146*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_283458942!
dense_1/StatefulPartitionedCall°
adv_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0adv_1_28346149adv_1_28346151*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_adv_1_layer_call_and_return_conditional_losses_283459112
adv_1/StatefulPartitionedCall°
val_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0val_1_28346154val_1_28346156*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_val_1_layer_call_and_return_conditional_losses_283459282
val_1/StatefulPartitionedCall®
adv_2/StatefulPartitionedCallStatefulPartitionedCall&adv_1/StatefulPartitionedCall:output:0adv_2_28346159adv_2_28346161*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_adv_2_layer_call_and_return_conditional_losses_283459452
adv_2/StatefulPartitionedCall®
val_2/StatefulPartitionedCallStatefulPartitionedCall&val_1/StatefulPartitionedCall:output:0val_2_28346164val_2_28346166*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_val_2_layer_call_and_return_conditional_losses_283459622
val_2/StatefulPartitionedCall®
adv_3/StatefulPartitionedCallStatefulPartitionedCall&adv_2/StatefulPartitionedCall:output:0adv_3_28346169adv_3_28346171*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_adv_3_layer_call_and_return_conditional_losses_283459782
adv_3/StatefulPartitionedCall®
val_3/StatefulPartitionedCallStatefulPartitionedCall&val_2/StatefulPartitionedCall:output:0val_3_28346174val_3_28346176*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_val_3_layer_call_and_return_conditional_losses_283459942
val_3/StatefulPartitionedCallÇ
tf.__operators__.add_1/AddV2AddV2&val_3/StatefulPartitionedCall:output:0&adv_3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_1/AddV2
,tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,tf.math.reduce_mean_1/Mean/reduction_indicesâ
tf.math.reduce_mean_1/MeanMean&adv_3/StatefulPartitionedCall:output:05tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
tf.math.reduce_mean_1/Mean°
tf.math.subtract_1/SubSub tf.__operators__.add_1/AddV2:z:0#tf.math.reduce_mean_1/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.subtract_1/SubÐ
IdentityIdentitytf.math.subtract_1/Sub:z:0^adv_1/StatefulPartitionedCall^adv_2/StatefulPartitionedCall^adv_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^val_1/StatefulPartitionedCall^val_2/StatefulPartitionedCall^val_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ?: : : : : : : : : : : : : : 2>
adv_1/StatefulPartitionedCalladv_1/StatefulPartitionedCall2>
adv_2/StatefulPartitionedCalladv_2/StatefulPartitionedCall2>
adv_3/StatefulPartitionedCalladv_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
val_1/StatefulPartitionedCallval_1/StatefulPartitionedCall2>
val_2/StatefulPartitionedCallval_2/StatefulPartitionedCall2>
val_3/StatefulPartitionedCallval_3/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?
 
_user_specified_nameinputs


(__inference_adv_2_layer_call_fn_28346645

inputs
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_adv_2_layer_call_and_return_conditional_losses_283459452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

*__inference_dense_1_layer_call_fn_28346565

inputs
unknown:	?
	unknown_0:	
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_283458942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ?: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?
 
_user_specified_nameinputs


(__inference_adv_3_layer_call_fn_28346683

inputs
unknown:@
	unknown_0:
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_adv_3_layer_call_and_return_conditional_losses_283459782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ÌC


E__inference_model_1_layer_call_and_return_conditional_losses_28346479

inputs9
&dense_1_matmul_readvariableop_resource:	?6
'dense_1_biasadd_readvariableop_resource:	7
$adv_1_matmul_readvariableop_resource:	@3
%adv_1_biasadd_readvariableop_resource:@7
$val_1_matmul_readvariableop_resource:	@3
%val_1_biasadd_readvariableop_resource:@6
$adv_2_matmul_readvariableop_resource:@@3
%adv_2_biasadd_readvariableop_resource:@6
$val_2_matmul_readvariableop_resource:@@3
%val_2_biasadd_readvariableop_resource:@6
$adv_3_matmul_readvariableop_resource:@3
%adv_3_biasadd_readvariableop_resource:6
$val_3_matmul_readvariableop_resource:@3
%val_3_biasadd_readvariableop_resource:
identity¢adv_1/BiasAdd/ReadVariableOp¢adv_1/MatMul/ReadVariableOp¢adv_2/BiasAdd/ReadVariableOp¢adv_2/MatMul/ReadVariableOp¢adv_3/BiasAdd/ReadVariableOp¢adv_3/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢val_1/BiasAdd/ReadVariableOp¢val_1/MatMul/ReadVariableOp¢val_2/BiasAdd/ReadVariableOp¢val_2/MatMul/ReadVariableOp¢val_3/BiasAdd/ReadVariableOp¢val_3/MatMul/ReadVariableOp¦
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/MatMul¥
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp¢
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Relu 
adv_1/MatMul/ReadVariableOpReadVariableOp$adv_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
adv_1/MatMul/ReadVariableOp
adv_1/MatMulMatMuldense_1/Relu:activations:0#adv_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_1/MatMul
adv_1/BiasAdd/ReadVariableOpReadVariableOp%adv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
adv_1/BiasAdd/ReadVariableOp
adv_1/BiasAddBiasAddadv_1/MatMul:product:0$adv_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_1/BiasAddj

adv_1/ReluReluadv_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

adv_1/Relu 
val_1/MatMul/ReadVariableOpReadVariableOp$val_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
val_1/MatMul/ReadVariableOp
val_1/MatMulMatMuldense_1/Relu:activations:0#val_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_1/MatMul
val_1/BiasAdd/ReadVariableOpReadVariableOp%val_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
val_1/BiasAdd/ReadVariableOp
val_1/BiasAddBiasAddval_1/MatMul:product:0$val_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_1/BiasAddj

val_1/ReluReluval_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

val_1/Relu
adv_2/MatMul/ReadVariableOpReadVariableOp$adv_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
adv_2/MatMul/ReadVariableOp
adv_2/MatMulMatMuladv_1/Relu:activations:0#adv_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_2/MatMul
adv_2/BiasAdd/ReadVariableOpReadVariableOp%adv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
adv_2/BiasAdd/ReadVariableOp
adv_2/BiasAddBiasAddadv_2/MatMul:product:0$adv_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_2/BiasAddj

adv_2/ReluReluadv_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

adv_2/Relu
val_2/MatMul/ReadVariableOpReadVariableOp$val_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
val_2/MatMul/ReadVariableOp
val_2/MatMulMatMulval_1/Relu:activations:0#val_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_2/MatMul
val_2/BiasAdd/ReadVariableOpReadVariableOp%val_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
val_2/BiasAdd/ReadVariableOp
val_2/BiasAddBiasAddval_2/MatMul:product:0$val_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_2/BiasAddj

val_2/ReluReluval_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

val_2/Relu
adv_3/MatMul/ReadVariableOpReadVariableOp$adv_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
adv_3/MatMul/ReadVariableOp
adv_3/MatMulMatMuladv_2/Relu:activations:0#adv_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adv_3/MatMul
adv_3/BiasAdd/ReadVariableOpReadVariableOp%adv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
adv_3/BiasAdd/ReadVariableOp
adv_3/BiasAddBiasAddadv_3/MatMul:product:0$adv_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adv_3/BiasAdd
val_3/MatMul/ReadVariableOpReadVariableOp$val_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
val_3/MatMul/ReadVariableOp
val_3/MatMulMatMulval_2/Relu:activations:0#val_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
val_3/MatMul
val_3/BiasAdd/ReadVariableOpReadVariableOp%val_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
val_3/BiasAdd/ReadVariableOp
val_3/BiasAddBiasAddval_3/MatMul:product:0$val_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
val_3/BiasAdd§
tf.__operators__.add_1/AddV2AddV2val_3/BiasAdd:output:0adv_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_1/AddV2
,tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,tf.math.reduce_mean_1/Mean/reduction_indicesÒ
tf.math.reduce_mean_1/MeanMeanadv_3/BiasAdd:output:05tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
tf.math.reduce_mean_1/Mean°
tf.math.subtract_1/SubSub tf.__operators__.add_1/AddV2:z:0#tf.math.reduce_mean_1/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.subtract_1/Sub
IdentityIdentitytf.math.subtract_1/Sub:z:0^adv_1/BiasAdd/ReadVariableOp^adv_1/MatMul/ReadVariableOp^adv_2/BiasAdd/ReadVariableOp^adv_2/MatMul/ReadVariableOp^adv_3/BiasAdd/ReadVariableOp^adv_3/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^val_1/BiasAdd/ReadVariableOp^val_1/MatMul/ReadVariableOp^val_2/BiasAdd/ReadVariableOp^val_2/MatMul/ReadVariableOp^val_3/BiasAdd/ReadVariableOp^val_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ?: : : : : : : : : : : : : : 2<
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
:ÿÿÿÿÿÿÿÿÿ?
 
_user_specified_nameinputs
«

ô
C__inference_val_2_layer_call_and_return_conditional_losses_28346616

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
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
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ú)
¦
!__inference__traced_save_28346754
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
%savev2_adv_3_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¡
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesª
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÂ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop'savev2_val_1_kernel_read_readvariableop%savev2_val_1_bias_read_readvariableop'savev2_adv_1_kernel_read_readvariableop%savev2_adv_1_bias_read_readvariableop'savev2_val_2_kernel_read_readvariableop%savev2_val_2_bias_read_readvariableop'savev2_adv_2_kernel_read_readvariableop%savev2_adv_2_bias_read_readvariableop'savev2_val_3_kernel_read_readvariableop%savev2_val_3_bias_read_readvariableop'savev2_adv_3_kernel_read_readvariableop%savev2_adv_3_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*
_input_shapes~
|: :	?::	@:@:	@:@:@@:@:@@:@:@::@:: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?:!

_output_shapes	
::%!

_output_shapes
:	@: 

_output_shapes
:@:%!

_output_shapes
:	@: 
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
: 
Ï	
ô
C__inference_val_3_layer_call_and_return_conditional_losses_28346655

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
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
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


(__inference_val_1_layer_call_fn_28346585

inputs
unknown:	@
	unknown_0:@
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_val_1_layer_call_and_return_conditional_losses_283459282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±
ç
*__inference_model_1_layer_call_fn_28346036
observation
unknown:	?
	unknown_0:	
	unknown_1:	@
	unknown_2:@
	unknown_3:	@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:

unknown_11:@

unknown_12:
identity¢StatefulPartitionedCall
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
GPU2*0J 8 *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_283460052
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
«

ô
C__inference_val_2_layer_call_and_return_conditional_losses_28345962

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
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
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¯

õ
C__inference_val_1_layer_call_and_return_conditional_losses_28346576

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
£E
	
$__inference__traced_restore_28346812
file_prefix2
assignvariableop_dense_1_kernel:	?.
assignvariableop_1_dense_1_bias:	2
assignvariableop_2_val_1_kernel:	@+
assignvariableop_3_val_1_bias:@2
assignvariableop_4_adv_1_kernel:	@+
assignvariableop_5_adv_1_bias:@1
assignvariableop_6_val_2_kernel:@@+
assignvariableop_7_val_2_bias:@1
assignvariableop_8_adv_2_kernel:@@+
assignvariableop_9_adv_2_bias:@2
 assignvariableop_10_val_3_kernel:@,
assignvariableop_11_val_3_bias:2
 assignvariableop_12_adv_3_kernel:@,
assignvariableop_13_adv_3_bias:#
assignvariableop_14_total: #
assignvariableop_15_count: 
identity_17¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¡
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names°
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*X
_output_shapesF
D:::::::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¤
AssignVariableOp_2AssignVariableOpassignvariableop_2_val_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¢
AssignVariableOp_3AssignVariableOpassignvariableop_3_val_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¤
AssignVariableOp_4AssignVariableOpassignvariableop_4_adv_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¢
AssignVariableOp_5AssignVariableOpassignvariableop_5_adv_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¤
AssignVariableOp_6AssignVariableOpassignvariableop_6_val_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¢
AssignVariableOp_7AssignVariableOpassignvariableop_7_val_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¤
AssignVariableOp_8AssignVariableOpassignvariableop_8_adv_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¢
AssignVariableOp_9AssignVariableOpassignvariableop_9_adv_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¨
AssignVariableOp_10AssignVariableOp assignvariableop_10_val_3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¦
AssignVariableOp_11AssignVariableOpassignvariableop_11_val_3_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¨
AssignVariableOp_12AssignVariableOp assignvariableop_12_adv_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¦
AssignVariableOp_13AssignVariableOpassignvariableop_13_adv_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¡
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¡
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_159
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp¾
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_16±
Identity_17IdentityIdentity_16:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_17"#
identity_17Identity_17:output:0*5
_input_shapes$
": : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
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
µ

ø
E__inference_dense_1_layer_call_and_return_conditional_losses_28346556

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
Þ+

E__inference_model_1_layer_call_and_return_conditional_losses_28346334
observation#
dense_1_28346294:	?
dense_1_28346296:	!
adv_1_28346299:	@
adv_1_28346301:@!
val_1_28346304:	@
val_1_28346306:@ 
adv_2_28346309:@@
adv_2_28346311:@ 
val_2_28346314:@@
val_2_28346316:@ 
adv_3_28346319:@
adv_3_28346321: 
val_3_28346324:@
val_3_28346326:
identity¢adv_1/StatefulPartitionedCall¢adv_2/StatefulPartitionedCall¢adv_3/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢val_1/StatefulPartitionedCall¢val_2/StatefulPartitionedCall¢val_3/StatefulPartitionedCall
dense_1/StatefulPartitionedCallStatefulPartitionedCallobservationdense_1_28346294dense_1_28346296*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_283458942!
dense_1/StatefulPartitionedCall°
adv_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0adv_1_28346299adv_1_28346301*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_adv_1_layer_call_and_return_conditional_losses_283459112
adv_1/StatefulPartitionedCall°
val_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0val_1_28346304val_1_28346306*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_val_1_layer_call_and_return_conditional_losses_283459282
val_1/StatefulPartitionedCall®
adv_2/StatefulPartitionedCallStatefulPartitionedCall&adv_1/StatefulPartitionedCall:output:0adv_2_28346309adv_2_28346311*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_adv_2_layer_call_and_return_conditional_losses_283459452
adv_2/StatefulPartitionedCall®
val_2/StatefulPartitionedCallStatefulPartitionedCall&val_1/StatefulPartitionedCall:output:0val_2_28346314val_2_28346316*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_val_2_layer_call_and_return_conditional_losses_283459622
val_2/StatefulPartitionedCall®
adv_3/StatefulPartitionedCallStatefulPartitionedCall&adv_2/StatefulPartitionedCall:output:0adv_3_28346319adv_3_28346321*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_adv_3_layer_call_and_return_conditional_losses_283459782
adv_3/StatefulPartitionedCall®
val_3/StatefulPartitionedCallStatefulPartitionedCall&val_2/StatefulPartitionedCall:output:0val_3_28346324val_3_28346326*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_val_3_layer_call_and_return_conditional_losses_283459942
val_3/StatefulPartitionedCallÇ
tf.__operators__.add_1/AddV2AddV2&val_3/StatefulPartitionedCall:output:0&adv_3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_1/AddV2
,tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,tf.math.reduce_mean_1/Mean/reduction_indicesâ
tf.math.reduce_mean_1/MeanMean&adv_3/StatefulPartitionedCall:output:05tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
tf.math.reduce_mean_1/Mean°
tf.math.subtract_1/SubSub tf.__operators__.add_1/AddV2:z:0#tf.math.reduce_mean_1/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.subtract_1/SubÐ
IdentityIdentitytf.math.subtract_1/Sub:z:0^adv_1/StatefulPartitionedCall^adv_2/StatefulPartitionedCall^adv_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^val_1/StatefulPartitionedCall^val_2/StatefulPartitionedCall^val_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ?: : : : : : : : : : : : : : 2>
adv_1/StatefulPartitionedCalladv_1/StatefulPartitionedCall2>
adv_2/StatefulPartitionedCalladv_2/StatefulPartitionedCall2>
adv_3/StatefulPartitionedCalladv_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
val_1/StatefulPartitionedCallval_1/StatefulPartitionedCall2>
val_2/StatefulPartitionedCallval_2/StatefulPartitionedCall2>
val_3/StatefulPartitionedCallval_3/StatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?
%
_user_specified_nameobservation
µ

ø
E__inference_dense_1_layer_call_and_return_conditional_losses_28345894

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

ã
&__inference_signature_wrapper_28346369
observation
unknown:	?
	unknown_0:	
	unknown_1:	@
	unknown_2:@
	unknown_3:	@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:

unknown_11:@

unknown_12:
identity¢StatefulPartitionedCallú
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
GPU2*0J 8 *,
f'R%
#__inference__wrapped_model_283458762
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
±
ç
*__inference_model_1_layer_call_fn_28346248
observation
unknown:	?
	unknown_0:	
	unknown_1:	@
	unknown_2:@
	unknown_3:	@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:

unknown_11:@

unknown_12:
identity¢StatefulPartitionedCall
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
GPU2*0J 8 *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_283461842
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
Ï	
ô
C__inference_val_3_layer_call_and_return_conditional_losses_28345994

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
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
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ÌC


E__inference_model_1_layer_call_and_return_conditional_losses_28346424

inputs9
&dense_1_matmul_readvariableop_resource:	?6
'dense_1_biasadd_readvariableop_resource:	7
$adv_1_matmul_readvariableop_resource:	@3
%adv_1_biasadd_readvariableop_resource:@7
$val_1_matmul_readvariableop_resource:	@3
%val_1_biasadd_readvariableop_resource:@6
$adv_2_matmul_readvariableop_resource:@@3
%adv_2_biasadd_readvariableop_resource:@6
$val_2_matmul_readvariableop_resource:@@3
%val_2_biasadd_readvariableop_resource:@6
$adv_3_matmul_readvariableop_resource:@3
%adv_3_biasadd_readvariableop_resource:6
$val_3_matmul_readvariableop_resource:@3
%val_3_biasadd_readvariableop_resource:
identity¢adv_1/BiasAdd/ReadVariableOp¢adv_1/MatMul/ReadVariableOp¢adv_2/BiasAdd/ReadVariableOp¢adv_2/MatMul/ReadVariableOp¢adv_3/BiasAdd/ReadVariableOp¢adv_3/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢val_1/BiasAdd/ReadVariableOp¢val_1/MatMul/ReadVariableOp¢val_2/BiasAdd/ReadVariableOp¢val_2/MatMul/ReadVariableOp¢val_3/BiasAdd/ReadVariableOp¢val_3/MatMul/ReadVariableOp¦
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/MatMul¥
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp¢
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Relu 
adv_1/MatMul/ReadVariableOpReadVariableOp$adv_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
adv_1/MatMul/ReadVariableOp
adv_1/MatMulMatMuldense_1/Relu:activations:0#adv_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_1/MatMul
adv_1/BiasAdd/ReadVariableOpReadVariableOp%adv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
adv_1/BiasAdd/ReadVariableOp
adv_1/BiasAddBiasAddadv_1/MatMul:product:0$adv_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_1/BiasAddj

adv_1/ReluReluadv_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

adv_1/Relu 
val_1/MatMul/ReadVariableOpReadVariableOp$val_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
val_1/MatMul/ReadVariableOp
val_1/MatMulMatMuldense_1/Relu:activations:0#val_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_1/MatMul
val_1/BiasAdd/ReadVariableOpReadVariableOp%val_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
val_1/BiasAdd/ReadVariableOp
val_1/BiasAddBiasAddval_1/MatMul:product:0$val_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_1/BiasAddj

val_1/ReluReluval_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

val_1/Relu
adv_2/MatMul/ReadVariableOpReadVariableOp$adv_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
adv_2/MatMul/ReadVariableOp
adv_2/MatMulMatMuladv_1/Relu:activations:0#adv_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_2/MatMul
adv_2/BiasAdd/ReadVariableOpReadVariableOp%adv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
adv_2/BiasAdd/ReadVariableOp
adv_2/BiasAddBiasAddadv_2/MatMul:product:0$adv_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
adv_2/BiasAddj

adv_2/ReluReluadv_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

adv_2/Relu
val_2/MatMul/ReadVariableOpReadVariableOp$val_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
val_2/MatMul/ReadVariableOp
val_2/MatMulMatMulval_1/Relu:activations:0#val_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_2/MatMul
val_2/BiasAdd/ReadVariableOpReadVariableOp%val_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
val_2/BiasAdd/ReadVariableOp
val_2/BiasAddBiasAddval_2/MatMul:product:0$val_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
val_2/BiasAddj

val_2/ReluReluval_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

val_2/Relu
adv_3/MatMul/ReadVariableOpReadVariableOp$adv_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
adv_3/MatMul/ReadVariableOp
adv_3/MatMulMatMuladv_2/Relu:activations:0#adv_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adv_3/MatMul
adv_3/BiasAdd/ReadVariableOpReadVariableOp%adv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
adv_3/BiasAdd/ReadVariableOp
adv_3/BiasAddBiasAddadv_3/MatMul:product:0$adv_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adv_3/BiasAdd
val_3/MatMul/ReadVariableOpReadVariableOp$val_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
val_3/MatMul/ReadVariableOp
val_3/MatMulMatMulval_2/Relu:activations:0#val_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
val_3/MatMul
val_3/BiasAdd/ReadVariableOpReadVariableOp%val_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
val_3/BiasAdd/ReadVariableOp
val_3/BiasAddBiasAddval_3/MatMul:product:0$val_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
val_3/BiasAdd§
tf.__operators__.add_1/AddV2AddV2val_3/BiasAdd:output:0adv_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_1/AddV2
,tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,tf.math.reduce_mean_1/Mean/reduction_indicesÒ
tf.math.reduce_mean_1/MeanMeanadv_3/BiasAdd:output:05tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
tf.math.reduce_mean_1/Mean°
tf.math.subtract_1/SubSub tf.__operators__.add_1/AddV2:z:0#tf.math.reduce_mean_1/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.subtract_1/Sub
IdentityIdentitytf.math.subtract_1/Sub:z:0^adv_1/BiasAdd/ReadVariableOp^adv_1/MatMul/ReadVariableOp^adv_2/BiasAdd/ReadVariableOp^adv_2/MatMul/ReadVariableOp^adv_3/BiasAdd/ReadVariableOp^adv_3/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^val_1/BiasAdd/ReadVariableOp^val_1/MatMul/ReadVariableOp^val_2/BiasAdd/ReadVariableOp^val_2/MatMul/ReadVariableOp^val_3/BiasAdd/ReadVariableOp^val_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ?: : : : : : : : : : : : : : 2<
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
:ÿÿÿÿÿÿÿÿÿ?
 
_user_specified_nameinputs
Ï+

E__inference_model_1_layer_call_and_return_conditional_losses_28346005

inputs#
dense_1_28345895:	?
dense_1_28345897:	!
adv_1_28345912:	@
adv_1_28345914:@!
val_1_28345929:	@
val_1_28345931:@ 
adv_2_28345946:@@
adv_2_28345948:@ 
val_2_28345963:@@
val_2_28345965:@ 
adv_3_28345979:@
adv_3_28345981: 
val_3_28345995:@
val_3_28345997:
identity¢adv_1/StatefulPartitionedCall¢adv_2/StatefulPartitionedCall¢adv_3/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢val_1/StatefulPartitionedCall¢val_2/StatefulPartitionedCall¢val_3/StatefulPartitionedCall
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_28345895dense_1_28345897*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_283458942!
dense_1/StatefulPartitionedCall°
adv_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0adv_1_28345912adv_1_28345914*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_adv_1_layer_call_and_return_conditional_losses_283459112
adv_1/StatefulPartitionedCall°
val_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0val_1_28345929val_1_28345931*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_val_1_layer_call_and_return_conditional_losses_283459282
val_1/StatefulPartitionedCall®
adv_2/StatefulPartitionedCallStatefulPartitionedCall&adv_1/StatefulPartitionedCall:output:0adv_2_28345946adv_2_28345948*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_adv_2_layer_call_and_return_conditional_losses_283459452
adv_2/StatefulPartitionedCall®
val_2/StatefulPartitionedCallStatefulPartitionedCall&val_1/StatefulPartitionedCall:output:0val_2_28345963val_2_28345965*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_val_2_layer_call_and_return_conditional_losses_283459622
val_2/StatefulPartitionedCall®
adv_3/StatefulPartitionedCallStatefulPartitionedCall&adv_2/StatefulPartitionedCall:output:0adv_3_28345979adv_3_28345981*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_adv_3_layer_call_and_return_conditional_losses_283459782
adv_3/StatefulPartitionedCall®
val_3/StatefulPartitionedCallStatefulPartitionedCall&val_2/StatefulPartitionedCall:output:0val_3_28345995val_3_28345997*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_val_3_layer_call_and_return_conditional_losses_283459942
val_3/StatefulPartitionedCallÇ
tf.__operators__.add_1/AddV2AddV2&val_3/StatefulPartitionedCall:output:0&adv_3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_1/AddV2
,tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,tf.math.reduce_mean_1/Mean/reduction_indicesâ
tf.math.reduce_mean_1/MeanMean&adv_3/StatefulPartitionedCall:output:05tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
tf.math.reduce_mean_1/Mean°
tf.math.subtract_1/SubSub tf.__operators__.add_1/AddV2:z:0#tf.math.reduce_mean_1/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.subtract_1/SubÐ
IdentityIdentitytf.math.subtract_1/Sub:z:0^adv_1/StatefulPartitionedCall^adv_2/StatefulPartitionedCall^adv_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^val_1/StatefulPartitionedCall^val_2/StatefulPartitionedCall^val_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ?: : : : : : : : : : : : : : 2>
adv_1/StatefulPartitionedCalladv_1/StatefulPartitionedCall2>
adv_2/StatefulPartitionedCalladv_2/StatefulPartitionedCall2>
adv_3/StatefulPartitionedCalladv_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
val_1/StatefulPartitionedCallval_1/StatefulPartitionedCall2>
val_2/StatefulPartitionedCallval_2/StatefulPartitionedCall2>
val_3/StatefulPartitionedCallval_3/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?
 
_user_specified_nameinputs


(__inference_adv_1_layer_call_fn_28346605

inputs
unknown:	@
	unknown_0:@
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_adv_1_layer_call_and_return_conditional_losses_283459112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æO
Ù
#__inference__wrapped_model_28345876
observationA
.model_1_dense_1_matmul_readvariableop_resource:	?>
/model_1_dense_1_biasadd_readvariableop_resource:	?
,model_1_adv_1_matmul_readvariableop_resource:	@;
-model_1_adv_1_biasadd_readvariableop_resource:@?
,model_1_val_1_matmul_readvariableop_resource:	@;
-model_1_val_1_biasadd_readvariableop_resource:@>
,model_1_adv_2_matmul_readvariableop_resource:@@;
-model_1_adv_2_biasadd_readvariableop_resource:@>
,model_1_val_2_matmul_readvariableop_resource:@@;
-model_1_val_2_biasadd_readvariableop_resource:@>
,model_1_adv_3_matmul_readvariableop_resource:@;
-model_1_adv_3_biasadd_readvariableop_resource:>
,model_1_val_3_matmul_readvariableop_resource:@;
-model_1_val_3_biasadd_readvariableop_resource:
identity¢$model_1/adv_1/BiasAdd/ReadVariableOp¢#model_1/adv_1/MatMul/ReadVariableOp¢$model_1/adv_2/BiasAdd/ReadVariableOp¢#model_1/adv_2/MatMul/ReadVariableOp¢$model_1/adv_3/BiasAdd/ReadVariableOp¢#model_1/adv_3/MatMul/ReadVariableOp¢&model_1/dense_1/BiasAdd/ReadVariableOp¢%model_1/dense_1/MatMul/ReadVariableOp¢$model_1/val_1/BiasAdd/ReadVariableOp¢#model_1/val_1/MatMul/ReadVariableOp¢$model_1/val_2/BiasAdd/ReadVariableOp¢#model_1/val_2/MatMul/ReadVariableOp¢$model_1/val_3/BiasAdd/ReadVariableOp¢#model_1/val_3/MatMul/ReadVariableOp¾
%model_1/dense_1/MatMul/ReadVariableOpReadVariableOp.model_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02'
%model_1/dense_1/MatMul/ReadVariableOp©
model_1/dense_1/MatMulMatMulobservation-model_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense_1/MatMul½
&model_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02(
&model_1/dense_1/BiasAdd/ReadVariableOpÂ
model_1/dense_1/BiasAddBiasAdd model_1/dense_1/MatMul:product:0.model_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense_1/BiasAdd
model_1/dense_1/ReluRelu model_1/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense_1/Relu¸
#model_1/adv_1/MatMul/ReadVariableOpReadVariableOp,model_1_adv_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02%
#model_1/adv_1/MatMul/ReadVariableOp¹
model_1/adv_1/MatMulMatMul"model_1/dense_1/Relu:activations:0+model_1/adv_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/adv_1/MatMul¶
$model_1/adv_1/BiasAdd/ReadVariableOpReadVariableOp-model_1_adv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$model_1/adv_1/BiasAdd/ReadVariableOp¹
model_1/adv_1/BiasAddBiasAddmodel_1/adv_1/MatMul:product:0,model_1/adv_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/adv_1/BiasAdd
model_1/adv_1/ReluRelumodel_1/adv_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/adv_1/Relu¸
#model_1/val_1/MatMul/ReadVariableOpReadVariableOp,model_1_val_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02%
#model_1/val_1/MatMul/ReadVariableOp¹
model_1/val_1/MatMulMatMul"model_1/dense_1/Relu:activations:0+model_1/val_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/val_1/MatMul¶
$model_1/val_1/BiasAdd/ReadVariableOpReadVariableOp-model_1_val_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$model_1/val_1/BiasAdd/ReadVariableOp¹
model_1/val_1/BiasAddBiasAddmodel_1/val_1/MatMul:product:0,model_1/val_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/val_1/BiasAdd
model_1/val_1/ReluRelumodel_1/val_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/val_1/Relu·
#model_1/adv_2/MatMul/ReadVariableOpReadVariableOp,model_1_adv_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02%
#model_1/adv_2/MatMul/ReadVariableOp·
model_1/adv_2/MatMulMatMul model_1/adv_1/Relu:activations:0+model_1/adv_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/adv_2/MatMul¶
$model_1/adv_2/BiasAdd/ReadVariableOpReadVariableOp-model_1_adv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$model_1/adv_2/BiasAdd/ReadVariableOp¹
model_1/adv_2/BiasAddBiasAddmodel_1/adv_2/MatMul:product:0,model_1/adv_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/adv_2/BiasAdd
model_1/adv_2/ReluRelumodel_1/adv_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/adv_2/Relu·
#model_1/val_2/MatMul/ReadVariableOpReadVariableOp,model_1_val_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02%
#model_1/val_2/MatMul/ReadVariableOp·
model_1/val_2/MatMulMatMul model_1/val_1/Relu:activations:0+model_1/val_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/val_2/MatMul¶
$model_1/val_2/BiasAdd/ReadVariableOpReadVariableOp-model_1_val_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$model_1/val_2/BiasAdd/ReadVariableOp¹
model_1/val_2/BiasAddBiasAddmodel_1/val_2/MatMul:product:0,model_1/val_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/val_2/BiasAdd
model_1/val_2/ReluRelumodel_1/val_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/val_2/Relu·
#model_1/adv_3/MatMul/ReadVariableOpReadVariableOp,model_1_adv_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02%
#model_1/adv_3/MatMul/ReadVariableOp·
model_1/adv_3/MatMulMatMul model_1/adv_2/Relu:activations:0+model_1/adv_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/adv_3/MatMul¶
$model_1/adv_3/BiasAdd/ReadVariableOpReadVariableOp-model_1_adv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model_1/adv_3/BiasAdd/ReadVariableOp¹
model_1/adv_3/BiasAddBiasAddmodel_1/adv_3/MatMul:product:0,model_1/adv_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/adv_3/BiasAdd·
#model_1/val_3/MatMul/ReadVariableOpReadVariableOp,model_1_val_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02%
#model_1/val_3/MatMul/ReadVariableOp·
model_1/val_3/MatMulMatMul model_1/val_2/Relu:activations:0+model_1/val_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/val_3/MatMul¶
$model_1/val_3/BiasAdd/ReadVariableOpReadVariableOp-model_1_val_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model_1/val_3/BiasAdd/ReadVariableOp¹
model_1/val_3/BiasAddBiasAddmodel_1/val_3/MatMul:product:0,model_1/val_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/val_3/BiasAddÇ
$model_1/tf.__operators__.add_1/AddV2AddV2model_1/val_3/BiasAdd:output:0model_1/adv_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$model_1/tf.__operators__.add_1/AddV2®
4model_1/tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :26
4model_1/tf.math.reduce_mean_1/Mean/reduction_indicesò
"model_1/tf.math.reduce_mean_1/MeanMeanmodel_1/adv_3/BiasAdd:output:0=model_1/tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2$
"model_1/tf.math.reduce_mean_1/MeanÐ
model_1/tf.math.subtract_1/SubSub(model_1/tf.__operators__.add_1/AddV2:z:0+model_1/tf.math.reduce_mean_1/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
model_1/tf.math.subtract_1/Sub
IdentityIdentity"model_1/tf.math.subtract_1/Sub:z:0%^model_1/adv_1/BiasAdd/ReadVariableOp$^model_1/adv_1/MatMul/ReadVariableOp%^model_1/adv_2/BiasAdd/ReadVariableOp$^model_1/adv_2/MatMul/ReadVariableOp%^model_1/adv_3/BiasAdd/ReadVariableOp$^model_1/adv_3/MatMul/ReadVariableOp'^model_1/dense_1/BiasAdd/ReadVariableOp&^model_1/dense_1/MatMul/ReadVariableOp%^model_1/val_1/BiasAdd/ReadVariableOp$^model_1/val_1/MatMul/ReadVariableOp%^model_1/val_2/BiasAdd/ReadVariableOp$^model_1/val_2/MatMul/ReadVariableOp%^model_1/val_3/BiasAdd/ReadVariableOp$^model_1/val_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ?: : : : : : : : : : : : : : 2L
$model_1/adv_1/BiasAdd/ReadVariableOp$model_1/adv_1/BiasAdd/ReadVariableOp2J
#model_1/adv_1/MatMul/ReadVariableOp#model_1/adv_1/MatMul/ReadVariableOp2L
$model_1/adv_2/BiasAdd/ReadVariableOp$model_1/adv_2/BiasAdd/ReadVariableOp2J
#model_1/adv_2/MatMul/ReadVariableOp#model_1/adv_2/MatMul/ReadVariableOp2L
$model_1/adv_3/BiasAdd/ReadVariableOp$model_1/adv_3/BiasAdd/ReadVariableOp2J
#model_1/adv_3/MatMul/ReadVariableOp#model_1/adv_3/MatMul/ReadVariableOp2P
&model_1/dense_1/BiasAdd/ReadVariableOp&model_1/dense_1/BiasAdd/ReadVariableOp2N
%model_1/dense_1/MatMul/ReadVariableOp%model_1/dense_1/MatMul/ReadVariableOp2L
$model_1/val_1/BiasAdd/ReadVariableOp$model_1/val_1/BiasAdd/ReadVariableOp2J
#model_1/val_1/MatMul/ReadVariableOp#model_1/val_1/MatMul/ReadVariableOp2L
$model_1/val_2/BiasAdd/ReadVariableOp$model_1/val_2/BiasAdd/ReadVariableOp2J
#model_1/val_2/MatMul/ReadVariableOp#model_1/val_2/MatMul/ReadVariableOp2L
$model_1/val_3/BiasAdd/ReadVariableOp$model_1/val_3/BiasAdd/ReadVariableOp2J
#model_1/val_3/MatMul/ReadVariableOp#model_1/val_3/MatMul/ReadVariableOp:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?
%
_user_specified_nameobservation
¢
â
*__inference_model_1_layer_call_fn_28346545

inputs
unknown:	?
	unknown_0:	
	unknown_1:	@
	unknown_2:@
	unknown_3:	@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:

unknown_11:@

unknown_12:
identity¢StatefulPartitionedCall
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
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_283461842
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
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?
 
_user_specified_nameinputs
¢
â
*__inference_model_1_layer_call_fn_28346512

inputs
unknown:	?
	unknown_0:	
	unknown_1:	@
	unknown_2:@
	unknown_3:	@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:

unknown_11:@

unknown_12:
identity¢StatefulPartitionedCall
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
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_283460052
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
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?
 
_user_specified_nameinputs
¯

õ
C__inference_val_1_layer_call_and_return_conditional_losses_28345928

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
¯

õ
C__inference_adv_1_layer_call_and_return_conditional_losses_28346596

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
Ï	
ô
C__inference_adv_3_layer_call_and_return_conditional_losses_28345978

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
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
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ï	
ô
C__inference_adv_3_layer_call_and_return_conditional_losses_28346674

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
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
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¯

õ
C__inference_adv_1_layer_call_and_return_conditional_losses_28345911

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
«

ô
C__inference_adv_2_layer_call_and_return_conditional_losses_28346636

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
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
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
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
tf.math.subtract_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Æ
ÛX
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
*l&call_and_return_all_conditional_losses
m__call__
n_default_save_signature"ÊT
_tf_keras_network®T{"name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 63]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "observation"}, "name": "observation", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["observation", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "val_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "val_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "adv_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "adv_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "val_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "val_2", "inbound_nodes": [[["val_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "adv_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "adv_2", "inbound_nodes": [[["adv_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "val_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "val_3", "inbound_nodes": [[["val_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "adv_3", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "adv_3", "inbound_nodes": [[["adv_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_1", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_1", "inbound_nodes": [["val_3", 0, 0, {"y": ["adv_3", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean_1", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean_1", "inbound_nodes": [["adv_3", 0, 0, {"axis": 1, "keepdims": true}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_1", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract_1", "inbound_nodes": [["tf.__operators__.add_1", 0, 0, {"y": ["tf.math.reduce_mean_1", 0, 0], "name": null}]]}], "input_layers": [["observation", 0, 0]], "output_layers": [["tf.math.subtract_1", 0, 0]]}, "shared_object_id": 25, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 63]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 63]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 63]}, "float32", "observation"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 63]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "observation"}, "name": "observation", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["observation", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Dense", "config": {"name": "val_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "val_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "adv_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "adv_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dense", "config": {"name": "val_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "val_2", "inbound_nodes": [[["val_1", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Dense", "config": {"name": "adv_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "adv_2", "inbound_nodes": [[["adv_1", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "Dense", "config": {"name": "val_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "val_3", "inbound_nodes": [[["val_2", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "Dense", "config": {"name": "adv_3", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "adv_3", "inbound_nodes": [[["adv_2", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_1", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_1", "inbound_nodes": [["val_3", 0, 0, {"y": ["adv_3", 0, 0], "name": null}]], "shared_object_id": 22}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean_1", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean_1", "inbound_nodes": [["adv_3", 0, 0, {"axis": 1, "keepdims": true}]], "shared_object_id": 23}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_1", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract_1", "inbound_nodes": [["tf.__operators__.add_1", 0, 0, {"y": ["tf.math.reduce_mean_1", 0, 0], "name": null}]], "shared_object_id": 24}], "input_layers": [["observation", 0, 0]], "output_layers": [["tf.math.subtract_1", 0, 0]]}}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}, "shared_object_id": 27}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
ó"ð
_tf_keras_input_layerÐ{"class_name": "InputLayer", "name": "observation", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 63]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 63]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "observation"}}
þ

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*o&call_and_return_all_conditional_losses
p__call__"Ù
_tf_keras_layer¿{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["observation", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 63}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 63]}}
÷

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*q&call_and_return_all_conditional_losses
r__call__"Ò
_tf_keras_layer¸{"name": "val_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "val_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 29}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
÷

kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
*s&call_and_return_all_conditional_losses
t__call__"Ò
_tf_keras_layer¸{"name": "adv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "adv_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 30}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
ö

$kernel
%bias
&	variables
'regularization_losses
(trainable_variables
)	keras_api
*u&call_and_return_all_conditional_losses
v__call__"Ñ
_tf_keras_layer·{"name": "val_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "val_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["val_1", 0, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ö

*kernel
+bias
,	variables
-regularization_losses
.trainable_variables
/	keras_api
*w&call_and_return_all_conditional_losses
x__call__"Ñ
_tf_keras_layer·{"name": "adv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "adv_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["adv_1", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 32}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
÷

0kernel
1bias
2	variables
3regularization_losses
4trainable_variables
5	keras_api
*y&call_and_return_all_conditional_losses
z__call__"Ò
_tf_keras_layer¸{"name": "val_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "val_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["val_2", 0, 0, {}]]], "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
÷

6kernel
7bias
8	variables
9regularization_losses
:trainable_variables
;	keras_api
*{&call_and_return_all_conditional_losses
|__call__"Ò
_tf_keras_layer¸{"name": "adv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "adv_3", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["adv_2", 0, 0, {}]]], "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 34}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
Ø
<	keras_api"Æ
_tf_keras_layer¬{"name": "tf.__operators__.add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_1", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "inbound_nodes": [["val_3", 0, 0, {"y": ["adv_3", 0, 0], "name": null}]], "shared_object_id": 22}
Î
=	keras_api"¼
_tf_keras_layer¢{"name": "tf.math.reduce_mean_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean_1", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "inbound_nodes": [["adv_3", 0, 0, {"axis": 1, "keepdims": true}]], "shared_object_id": 23}
í
>	keras_api"Û
_tf_keras_layerÁ{"name": "tf.math.subtract_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_1", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "inbound_nodes": [["tf.__operators__.add_1", 0, 0, {"y": ["tf.math.reduce_mean_1", 0, 0], "name": null}]], "shared_object_id": 24}
"
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
Ê
?non_trainable_variables

@layers
Alayer_metrics
Blayer_regularization_losses
	variables
regularization_losses
trainable_variables
Cmetrics
m__call__
n_default_save_signature
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
,
}serving_default"
signature_map
!:	?2dense_1/kernel
:2dense_1/bias
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
­
Dnon_trainable_variables

Elayers
Flayer_metrics
Glayer_regularization_losses
	variables
regularization_losses
trainable_variables
Hmetrics
p__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
:	@2val_1/kernel
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
­
Inon_trainable_variables

Jlayers
Klayer_metrics
Llayer_regularization_losses
	variables
regularization_losses
trainable_variables
Mmetrics
r__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
:	@2adv_1/kernel
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
­
Nnon_trainable_variables

Olayers
Player_metrics
Qlayer_regularization_losses
 	variables
!regularization_losses
"trainable_variables
Rmetrics
t__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
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
­
Snon_trainable_variables

Tlayers
Ulayer_metrics
Vlayer_regularization_losses
&	variables
'regularization_losses
(trainable_variables
Wmetrics
v__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
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
­
Xnon_trainable_variables

Ylayers
Zlayer_metrics
[layer_regularization_losses
,	variables
-regularization_losses
.trainable_variables
\metrics
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
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
­
]non_trainable_variables

^layers
_layer_metrics
`layer_regularization_losses
2	variables
3regularization_losses
4trainable_variables
ametrics
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
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
­
bnon_trainable_variables

clayers
dlayer_metrics
elayer_regularization_losses
8	variables
9regularization_losses
:trainable_variables
fmetrics
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
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
g0"
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
Ô
	htotal
	icount
j	variables
k	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 35}
:  (2total
:  (2count
.
h0
i1"
trackable_list_wrapper
-
j	variables"
_generic_user_object
â2ß
E__inference_model_1_layer_call_and_return_conditional_losses_28346424
E__inference_model_1_layer_call_and_return_conditional_losses_28346479
E__inference_model_1_layer_call_and_return_conditional_losses_28346291
E__inference_model_1_layer_call_and_return_conditional_losses_28346334À
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
ö2ó
*__inference_model_1_layer_call_fn_28346036
*__inference_model_1_layer_call_fn_28346512
*__inference_model_1_layer_call_fn_28346545
*__inference_model_1_layer_call_fn_28346248À
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
å2â
#__inference__wrapped_model_28345876º
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
ï2ì
E__inference_dense_1_layer_call_and_return_conditional_losses_28346556¢
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
Ô2Ñ
*__inference_dense_1_layer_call_fn_28346565¢
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
í2ê
C__inference_val_1_layer_call_and_return_conditional_losses_28346576¢
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
Ò2Ï
(__inference_val_1_layer_call_fn_28346585¢
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
í2ê
C__inference_adv_1_layer_call_and_return_conditional_losses_28346596¢
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
Ò2Ï
(__inference_adv_1_layer_call_fn_28346605¢
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
í2ê
C__inference_val_2_layer_call_and_return_conditional_losses_28346616¢
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
Ò2Ï
(__inference_val_2_layer_call_fn_28346625¢
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
í2ê
C__inference_adv_2_layer_call_and_return_conditional_losses_28346636¢
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
Ò2Ï
(__inference_adv_2_layer_call_fn_28346645¢
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
í2ê
C__inference_val_3_layer_call_and_return_conditional_losses_28346655¢
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
Ò2Ï
(__inference_val_3_layer_call_fn_28346664¢
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
í2ê
C__inference_adv_3_layer_call_and_return_conditional_losses_28346674¢
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
Ò2Ï
(__inference_adv_3_layer_call_fn_28346683¢
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
ÑBÎ
&__inference_signature_wrapper_28346369observation"
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
 ·
#__inference__wrapped_model_28345876*+$%67014¢1
*¢'
%"
observationÿÿÿÿÿÿÿÿÿ?
ª "GªD
B
tf.math.subtract_1,)
tf.math.subtract_1ÿÿÿÿÿÿÿÿÿ¤
C__inference_adv_1_layer_call_and_return_conditional_losses_28346596]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 |
(__inference_adv_1_layer_call_fn_28346605P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@£
C__inference_adv_2_layer_call_and_return_conditional_losses_28346636\*+/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 {
(__inference_adv_2_layer_call_fn_28346645O*+/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@£
C__inference_adv_3_layer_call_and_return_conditional_losses_28346674\67/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_adv_3_layer_call_fn_28346683O67/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ¦
E__inference_dense_1_layer_call_and_return_conditional_losses_28346556]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ?
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
*__inference_dense_1_layer_call_fn_28346565P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ?
ª "ÿÿÿÿÿÿÿÿÿ¾
E__inference_model_1_layer_call_and_return_conditional_losses_28346291u*+$%6701<¢9
2¢/
%"
observationÿÿÿÿÿÿÿÿÿ?
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
E__inference_model_1_layer_call_and_return_conditional_losses_28346334u*+$%6701<¢9
2¢/
%"
observationÿÿÿÿÿÿÿÿÿ?
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¹
E__inference_model_1_layer_call_and_return_conditional_losses_28346424p*+$%67017¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ?
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¹
E__inference_model_1_layer_call_and_return_conditional_losses_28346479p*+$%67017¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ?
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_model_1_layer_call_fn_28346036h*+$%6701<¢9
2¢/
%"
observationÿÿÿÿÿÿÿÿÿ?
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_model_1_layer_call_fn_28346248h*+$%6701<¢9
2¢/
%"
observationÿÿÿÿÿÿÿÿÿ?
p

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_model_1_layer_call_fn_28346512c*+$%67017¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ?
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_model_1_layer_call_fn_28346545c*+$%67017¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ?
p

 
ª "ÿÿÿÿÿÿÿÿÿÉ
&__inference_signature_wrapper_28346369*+$%6701C¢@
¢ 
9ª6
4
observation%"
observationÿÿÿÿÿÿÿÿÿ?"GªD
B
tf.math.subtract_1,)
tf.math.subtract_1ÿÿÿÿÿÿÿÿÿ¤
C__inference_val_1_layer_call_and_return_conditional_losses_28346576]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 |
(__inference_val_1_layer_call_fn_28346585P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@£
C__inference_val_2_layer_call_and_return_conditional_losses_28346616\$%/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 {
(__inference_val_2_layer_call_fn_28346625O$%/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@£
C__inference_val_3_layer_call_and_return_conditional_losses_28346655\01/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_val_3_layer_call_fn_28346664O01/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ