??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
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
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.2.0-dlenv2v2.2.0-0-g2b96f368??
?
Dense_Layer_1_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameDense_Layer_1_30/kernel
?
+Dense_Layer_1_30/kernel/Read/ReadVariableOpReadVariableOpDense_Layer_1_30/kernel* 
_output_shapes
:
??*
dtype0
?
Dense_Layer_1_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameDense_Layer_1_30/bias
|
)Dense_Layer_1_30/bias/Read/ReadVariableOpReadVariableOpDense_Layer_1_30/bias*
_output_shapes	
:?*
dtype0
?
Dense_Layer_2_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*(
shared_nameDense_Layer_2_19/kernel
?
+Dense_Layer_2_19/kernel/Read/ReadVariableOpReadVariableOpDense_Layer_2_19/kernel*
_output_shapes
:	?d*
dtype0
?
Dense_Layer_2_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameDense_Layer_2_19/bias
{
)Dense_Layer_2_19/bias/Read/ReadVariableOpReadVariableOpDense_Layer_2_19/bias*
_output_shapes
:d*
dtype0
?
Logit_Probs_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*&
shared_nameLogit_Probs_30/kernel

)Logit_Probs_30/kernel/Read/ReadVariableOpReadVariableOpLogit_Probs_30/kernel*
_output_shapes

:d
*
dtype0
~
Logit_Probs_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameLogit_Probs_30/bias
w
'Logit_Probs_30/bias/Read/ReadVariableOpReadVariableOpLogit_Probs_30/bias*
_output_shapes
:
*
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
?
Adam/Dense_Layer_1_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*/
shared_name Adam/Dense_Layer_1_30/kernel/m
?
2Adam/Dense_Layer_1_30/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_Layer_1_30/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/Dense_Layer_1_30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_nameAdam/Dense_Layer_1_30/bias/m
?
0Adam/Dense_Layer_1_30/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_Layer_1_30/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/Dense_Layer_2_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*/
shared_name Adam/Dense_Layer_2_19/kernel/m
?
2Adam/Dense_Layer_2_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_Layer_2_19/kernel/m*
_output_shapes
:	?d*
dtype0
?
Adam/Dense_Layer_2_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*-
shared_nameAdam/Dense_Layer_2_19/bias/m
?
0Adam/Dense_Layer_2_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_Layer_2_19/bias/m*
_output_shapes
:d*
dtype0
?
Adam/Logit_Probs_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*-
shared_nameAdam/Logit_Probs_30/kernel/m
?
0Adam/Logit_Probs_30/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Logit_Probs_30/kernel/m*
_output_shapes

:d
*
dtype0
?
Adam/Logit_Probs_30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_nameAdam/Logit_Probs_30/bias/m
?
.Adam/Logit_Probs_30/bias/m/Read/ReadVariableOpReadVariableOpAdam/Logit_Probs_30/bias/m*
_output_shapes
:
*
dtype0
?
Adam/Dense_Layer_1_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*/
shared_name Adam/Dense_Layer_1_30/kernel/v
?
2Adam/Dense_Layer_1_30/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_Layer_1_30/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/Dense_Layer_1_30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_nameAdam/Dense_Layer_1_30/bias/v
?
0Adam/Dense_Layer_1_30/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_Layer_1_30/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/Dense_Layer_2_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*/
shared_name Adam/Dense_Layer_2_19/kernel/v
?
2Adam/Dense_Layer_2_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_Layer_2_19/kernel/v*
_output_shapes
:	?d*
dtype0
?
Adam/Dense_Layer_2_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*-
shared_nameAdam/Dense_Layer_2_19/bias/v
?
0Adam/Dense_Layer_2_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_Layer_2_19/bias/v*
_output_shapes
:d*
dtype0
?
Adam/Logit_Probs_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*-
shared_nameAdam/Logit_Probs_30/kernel/v
?
0Adam/Logit_Probs_30/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Logit_Probs_30/kernel/v*
_output_shapes

:d
*
dtype0
?
Adam/Logit_Probs_30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_nameAdam/Logit_Probs_30/bias/v
?
.Adam/Logit_Probs_30/bias/v/Read/ReadVariableOpReadVariableOpAdam/Logit_Probs_30/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
?+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?*
value?*B?* B?*
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
	optimizer
regularization_losses
trainable_variables
		variables

	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
 	variables
!	keras_api
?
"iter

#beta_1

$beta_2
	%decay
&learning_ratemQmRmSmTmUmVvWvXvYvZv[v\
 
*
0
1
2
3
4
5
*
0
1
2
3
4
5
?
regularization_losses

'layers
trainable_variables
		variables
(layer_regularization_losses
)layer_metrics
*metrics
+non_trainable_variables
 
ca
VARIABLE_VALUEDense_Layer_1_30/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEDense_Layer_1_30/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses

,layers
-layer_regularization_losses
trainable_variables
	variables
.layer_metrics
/metrics
0non_trainable_variables
 
 
 
?
regularization_losses

1layers
2layer_regularization_losses
trainable_variables
	variables
3layer_metrics
4metrics
5non_trainable_variables
ca
VARIABLE_VALUEDense_Layer_2_19/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEDense_Layer_2_19/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses

6layers
7layer_regularization_losses
trainable_variables
	variables
8layer_metrics
9metrics
:non_trainable_variables
a_
VARIABLE_VALUELogit_Probs_30/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUELogit_Probs_30/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses

;layers
<layer_regularization_losses
trainable_variables
 	variables
=layer_metrics
>metrics
?non_trainable_variables
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
#
0
1
2
3
4
 
 

@0
A1
B2
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
	Ctotal
	Dcount
E	variables
F	keras_api
D
	Gtotal
	Hcount
I
_fn_kwargs
J	variables
K	keras_api
D
	Ltotal
	Mcount
N
_fn_kwargs
O	variables
P	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

C0
D1

E	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

G0
H1

J	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

L0
M1

O	variables
??
VARIABLE_VALUEAdam/Dense_Layer_1_30/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Dense_Layer_1_30/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Dense_Layer_2_19/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Dense_Layer_2_19/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Logit_Probs_30/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/Logit_Probs_30/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Dense_Layer_1_30/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Dense_Layer_1_30/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Dense_Layer_2_19/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Dense_Layer_2_19/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Logit_Probs_30/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/Logit_Probs_30/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_InputPlaceholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_InputDense_Layer_1_30/kernelDense_Layer_1_30/biasDense_Layer_2_19/kernelDense_Layer_2_19/biasLogit_Probs_30/kernelLogit_Probs_30/bias*
Tin
	2*
Tout
2*'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*.
f)R'
%__inference_signature_wrapper_7860941
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+Dense_Layer_1_30/kernel/Read/ReadVariableOp)Dense_Layer_1_30/bias/Read/ReadVariableOp+Dense_Layer_2_19/kernel/Read/ReadVariableOp)Dense_Layer_2_19/bias/Read/ReadVariableOp)Logit_Probs_30/kernel/Read/ReadVariableOp'Logit_Probs_30/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp2Adam/Dense_Layer_1_30/kernel/m/Read/ReadVariableOp0Adam/Dense_Layer_1_30/bias/m/Read/ReadVariableOp2Adam/Dense_Layer_2_19/kernel/m/Read/ReadVariableOp0Adam/Dense_Layer_2_19/bias/m/Read/ReadVariableOp0Adam/Logit_Probs_30/kernel/m/Read/ReadVariableOp.Adam/Logit_Probs_30/bias/m/Read/ReadVariableOp2Adam/Dense_Layer_1_30/kernel/v/Read/ReadVariableOp0Adam/Dense_Layer_1_30/bias/v/Read/ReadVariableOp2Adam/Dense_Layer_2_19/kernel/v/Read/ReadVariableOp0Adam/Dense_Layer_2_19/bias/v/Read/ReadVariableOp0Adam/Logit_Probs_30/kernel/v/Read/ReadVariableOp.Adam/Logit_Probs_30/bias/v/Read/ReadVariableOpConst**
Tin#
!2	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*)
f$R"
 __inference__traced_save_7861232
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameDense_Layer_1_30/kernelDense_Layer_1_30/biasDense_Layer_2_19/kernelDense_Layer_2_19/biasLogit_Probs_30/kernelLogit_Probs_30/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/Dense_Layer_1_30/kernel/mAdam/Dense_Layer_1_30/bias/mAdam/Dense_Layer_2_19/kernel/mAdam/Dense_Layer_2_19/bias/mAdam/Logit_Probs_30/kernel/mAdam/Logit_Probs_30/bias/mAdam/Dense_Layer_1_30/kernel/vAdam/Dense_Layer_1_30/bias/vAdam/Dense_Layer_2_19/kernel/vAdam/Dense_Layer_2_19/bias/vAdam/Logit_Probs_30/kernel/vAdam/Logit_Probs_30/bias/v*)
Tin"
 2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*,
f'R%
#__inference__traced_restore_7861331??
?
?
+__inference_initModel_layer_call_fn_7861015

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_initModel_layer_call_and_return_conditional_losses_78608622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
J__inference_Dense_Layer_1_layer_call_and_return_conditional_losses_7861043

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?I
?
 __inference__traced_save_7861232
file_prefix6
2savev2_dense_layer_1_30_kernel_read_readvariableop4
0savev2_dense_layer_1_30_bias_read_readvariableop6
2savev2_dense_layer_2_19_kernel_read_readvariableop4
0savev2_dense_layer_2_19_bias_read_readvariableop4
0savev2_logit_probs_30_kernel_read_readvariableop2
.savev2_logit_probs_30_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop=
9savev2_adam_dense_layer_1_30_kernel_m_read_readvariableop;
7savev2_adam_dense_layer_1_30_bias_m_read_readvariableop=
9savev2_adam_dense_layer_2_19_kernel_m_read_readvariableop;
7savev2_adam_dense_layer_2_19_bias_m_read_readvariableop;
7savev2_adam_logit_probs_30_kernel_m_read_readvariableop9
5savev2_adam_logit_probs_30_bias_m_read_readvariableop=
9savev2_adam_dense_layer_1_30_kernel_v_read_readvariableop;
7savev2_adam_dense_layer_1_30_bias_v_read_readvariableop=
9savev2_adam_dense_layer_2_19_kernel_v_read_readvariableop;
7savev2_adam_dense_layer_2_19_bias_v_read_readvariableop;
7savev2_adam_logit_probs_30_kernel_v_read_readvariableop9
5savev2_adam_logit_probs_30_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
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
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_839916086cab41a4ac8892c20a1bf100/part2	
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
value	B :2

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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_dense_layer_1_30_kernel_read_readvariableop0savev2_dense_layer_1_30_bias_read_readvariableop2savev2_dense_layer_2_19_kernel_read_readvariableop0savev2_dense_layer_2_19_bias_read_readvariableop0savev2_logit_probs_30_kernel_read_readvariableop.savev2_logit_probs_30_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop9savev2_adam_dense_layer_1_30_kernel_m_read_readvariableop7savev2_adam_dense_layer_1_30_bias_m_read_readvariableop9savev2_adam_dense_layer_2_19_kernel_m_read_readvariableop7savev2_adam_dense_layer_2_19_bias_m_read_readvariableop7savev2_adam_logit_probs_30_kernel_m_read_readvariableop5savev2_adam_logit_probs_30_bias_m_read_readvariableop9savev2_adam_dense_layer_1_30_kernel_v_read_readvariableop7savev2_adam_dense_layer_1_30_bias_v_read_readvariableop9savev2_adam_dense_layer_2_19_kernel_v_read_readvariableop7savev2_adam_dense_layer_2_19_bias_v_read_readvariableop7savev2_adam_logit_probs_30_kernel_v_read_readvariableop5savev2_adam_logit_probs_30_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *+
dtypes!
2	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:	?d:d:d
:
: : : : : : : : : : : :
??:?:	?d:d:d
:
:
??:?:	?d:d:d
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?d: 

_output_shapes
:d:$ 

_output_shapes

:d
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?d: 

_output_shapes
:d:$ 

_output_shapes

:d
: 

_output_shapes
:
:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?d: 

_output_shapes
:d:$ 

_output_shapes

:d
: 

_output_shapes
:
:

_output_shapes
: 
?
?
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_7861109

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d:::O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
%__inference_signature_wrapper_7860941	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*+
f&R$
"__inference__wrapped_model_78607042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
(
_output_shapes
:??????????

_user_specified_nameInput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
F__inference_initModel_layer_call_and_return_conditional_losses_7860998

inputs0
,dense_layer_1_matmul_readvariableop_resource1
-dense_layer_1_biasadd_readvariableop_resource0
,dense_layer_2_matmul_readvariableop_resource1
-dense_layer_2_biasadd_readvariableop_resource.
*logit_probs_matmul_readvariableop_resource/
+logit_probs_biasadd_readvariableop_resource
identity??
#Dense_Layer_1/MatMul/ReadVariableOpReadVariableOp,dense_layer_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#Dense_Layer_1/MatMul/ReadVariableOp?
Dense_Layer_1/MatMulMatMulinputs+Dense_Layer_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_Layer_1/MatMul?
$Dense_Layer_1/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$Dense_Layer_1/BiasAdd/ReadVariableOp?
Dense_Layer_1/BiasAddBiasAddDense_Layer_1/MatMul:product:0,Dense_Layer_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_Layer_1/BiasAdd?
Dense_Layer_1/ReluReluDense_Layer_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Dense_Layer_1/Relu?
dropout_30/IdentityIdentity Dense_Layer_1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout_30/Identity?
#Dense_Layer_2/MatMul/ReadVariableOpReadVariableOp,dense_layer_2_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02%
#Dense_Layer_2/MatMul/ReadVariableOp?
Dense_Layer_2/MatMulMatMuldropout_30/Identity:output:0+Dense_Layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
Dense_Layer_2/MatMul?
$Dense_Layer_2/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02&
$Dense_Layer_2/BiasAdd/ReadVariableOp?
Dense_Layer_2/BiasAddBiasAddDense_Layer_2/MatMul:product:0,Dense_Layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
Dense_Layer_2/BiasAdd?
Dense_Layer_2/ReluReluDense_Layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Dense_Layer_2/Relu?
!Logit_Probs/MatMul/ReadVariableOpReadVariableOp*logit_probs_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02#
!Logit_Probs/MatMul/ReadVariableOp?
Logit_Probs/MatMulMatMul Dense_Layer_2/Relu:activations:0)Logit_Probs/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Logit_Probs/MatMul?
"Logit_Probs/BiasAdd/ReadVariableOpReadVariableOp+logit_probs_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02$
"Logit_Probs/BiasAdd/ReadVariableOp?
Logit_Probs/BiasAddBiasAddLogit_Probs/MatMul:product:0*Logit_Probs/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Logit_Probs/BiasAddp
IdentityIdentityLogit_Probs/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????:::::::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
#__inference__traced_restore_7861331
file_prefix,
(assignvariableop_dense_layer_1_30_kernel,
(assignvariableop_1_dense_layer_1_30_bias.
*assignvariableop_2_dense_layer_2_19_kernel,
(assignvariableop_3_dense_layer_2_19_bias,
(assignvariableop_4_logit_probs_30_kernel*
&assignvariableop_5_logit_probs_30_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate
assignvariableop_11_total
assignvariableop_12_count
assignvariableop_13_total_1
assignvariableop_14_count_1
assignvariableop_15_total_2
assignvariableop_16_count_26
2assignvariableop_17_adam_dense_layer_1_30_kernel_m4
0assignvariableop_18_adam_dense_layer_1_30_bias_m6
2assignvariableop_19_adam_dense_layer_2_19_kernel_m4
0assignvariableop_20_adam_dense_layer_2_19_bias_m4
0assignvariableop_21_adam_logit_probs_30_kernel_m2
.assignvariableop_22_adam_logit_probs_30_bias_m6
2assignvariableop_23_adam_dense_layer_1_30_kernel_v4
0assignvariableop_24_adam_dense_layer_1_30_bias_v6
2assignvariableop_25_adam_dense_layer_2_19_kernel_v4
0assignvariableop_26_adam_dense_layer_2_19_bias_v4
0assignvariableop_27_adam_logit_probs_30_kernel_v2
.assignvariableop_28_adam_logit_probs_30_bias_v
identity_30??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp(assignvariableop_dense_layer_1_30_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp(assignvariableop_1_dense_layer_1_30_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp*assignvariableop_2_dense_layer_2_19_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp(assignvariableop_3_dense_layer_2_19_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp(assignvariableop_4_logit_probs_30_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp&assignvariableop_5_logit_probs_30_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0	*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_2Identity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_2Identity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp2assignvariableop_17_adam_dense_layer_1_30_kernel_mIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp0assignvariableop_18_adam_dense_layer_1_30_bias_mIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp2assignvariableop_19_adam_dense_layer_2_19_kernel_mIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp0assignvariableop_20_adam_dense_layer_2_19_bias_mIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp0assignvariableop_21_adam_logit_probs_30_kernel_mIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp.assignvariableop_22_adam_logit_probs_30_bias_mIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp2assignvariableop_23_adam_dense_layer_1_30_kernel_vIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp0assignvariableop_24_adam_dense_layer_1_30_bias_vIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp2assignvariableop_25_adam_dense_layer_2_19_kernel_vIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp0assignvariableop_26_adam_dense_layer_2_19_bias_vIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp0assignvariableop_27_adam_logit_probs_30_kernel_vIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp.assignvariableop_28_adam_logit_probs_30_bias_vIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_29Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_29?
Identity_30IdentityIdentity_29:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_30"#
identity_30Identity_30:output:0*?
_input_shapesx
v: :::::::::::::::::::::::::::::2$
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
AssignVariableOp_28AssignVariableOp_282(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
+__inference_initModel_layer_call_fn_7860877	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_initModel_layer_call_and_return_conditional_losses_78608622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
(
_output_shapes
:??????????

_user_specified_nameInput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
+__inference_initModel_layer_call_fn_7860914	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_initModel_layer_call_and_return_conditional_losses_78608992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
(
_output_shapes
:??????????

_user_specified_nameInput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
J__inference_Dense_Layer_2_layer_call_and_return_conditional_losses_7860776

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
e
G__inference_dropout_30_layer_call_and_return_conditional_losses_7861069

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
G__inference_dropout_30_layer_call_and_return_conditional_losses_7861064

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
G__inference_dropout_30_layer_call_and_return_conditional_losses_7860747

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_Dense_Layer_1_layer_call_fn_7861052

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_Dense_Layer_1_layer_call_and_return_conditional_losses_78607192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
J__inference_Dense_Layer_2_layer_call_and_return_conditional_losses_7861090

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
F__inference_initModel_layer_call_and_return_conditional_losses_7860839	
input
dense_layer_1_7860822
dense_layer_1_7860824
dense_layer_2_7860828
dense_layer_2_7860830
logit_probs_7860833
logit_probs_7860835
identity??%Dense_Layer_1/StatefulPartitionedCall?%Dense_Layer_2/StatefulPartitionedCall?#Logit_Probs/StatefulPartitionedCall?
%Dense_Layer_1/StatefulPartitionedCallStatefulPartitionedCallinputdense_layer_1_7860822dense_layer_1_7860824*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_Dense_Layer_1_layer_call_and_return_conditional_losses_78607192'
%Dense_Layer_1/StatefulPartitionedCall?
dropout_30/PartitionedCallPartitionedCall.Dense_Layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_30_layer_call_and_return_conditional_losses_78607522
dropout_30/PartitionedCall?
%Dense_Layer_2/StatefulPartitionedCallStatefulPartitionedCall#dropout_30/PartitionedCall:output:0dense_layer_2_7860828dense_layer_2_7860830*
Tin
2*
Tout
2*'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_Dense_Layer_2_layer_call_and_return_conditional_losses_78607762'
%Dense_Layer_2/StatefulPartitionedCall?
#Logit_Probs/StatefulPartitionedCallStatefulPartitionedCall.Dense_Layer_2/StatefulPartitionedCall:output:0logit_probs_7860833logit_probs_7860835*
Tin
2*
Tout
2*'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*Q
fLRJ
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_78608022%
#Logit_Probs/StatefulPartitionedCall?
IdentityIdentity,Logit_Probs/StatefulPartitionedCall:output:0&^Dense_Layer_1/StatefulPartitionedCall&^Dense_Layer_2/StatefulPartitionedCall$^Logit_Probs/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::2N
%Dense_Layer_1/StatefulPartitionedCall%Dense_Layer_1/StatefulPartitionedCall2N
%Dense_Layer_2/StatefulPartitionedCall%Dense_Layer_2/StatefulPartitionedCall2J
#Logit_Probs/StatefulPartitionedCall#Logit_Probs/StatefulPartitionedCall:O K
(
_output_shapes
:??????????

_user_specified_nameInput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
F__inference_initModel_layer_call_and_return_conditional_losses_7860862

inputs
dense_layer_1_7860845
dense_layer_1_7860847
dense_layer_2_7860851
dense_layer_2_7860853
logit_probs_7860856
logit_probs_7860858
identity??%Dense_Layer_1/StatefulPartitionedCall?%Dense_Layer_2/StatefulPartitionedCall?#Logit_Probs/StatefulPartitionedCall?"dropout_30/StatefulPartitionedCall?
%Dense_Layer_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_layer_1_7860845dense_layer_1_7860847*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_Dense_Layer_1_layer_call_and_return_conditional_losses_78607192'
%Dense_Layer_1/StatefulPartitionedCall?
"dropout_30/StatefulPartitionedCallStatefulPartitionedCall.Dense_Layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_30_layer_call_and_return_conditional_losses_78607472$
"dropout_30/StatefulPartitionedCall?
%Dense_Layer_2/StatefulPartitionedCallStatefulPartitionedCall+dropout_30/StatefulPartitionedCall:output:0dense_layer_2_7860851dense_layer_2_7860853*
Tin
2*
Tout
2*'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_Dense_Layer_2_layer_call_and_return_conditional_losses_78607762'
%Dense_Layer_2/StatefulPartitionedCall?
#Logit_Probs/StatefulPartitionedCallStatefulPartitionedCall.Dense_Layer_2/StatefulPartitionedCall:output:0logit_probs_7860856logit_probs_7860858*
Tin
2*
Tout
2*'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*Q
fLRJ
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_78608022%
#Logit_Probs/StatefulPartitionedCall?
IdentityIdentity,Logit_Probs/StatefulPartitionedCall:output:0&^Dense_Layer_1/StatefulPartitionedCall&^Dense_Layer_2/StatefulPartitionedCall$^Logit_Probs/StatefulPartitionedCall#^dropout_30/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::2N
%Dense_Layer_1/StatefulPartitionedCall%Dense_Layer_1/StatefulPartitionedCall2N
%Dense_Layer_2/StatefulPartitionedCall%Dense_Layer_2/StatefulPartitionedCall2J
#Logit_Probs/StatefulPartitionedCall#Logit_Probs/StatefulPartitionedCall2H
"dropout_30/StatefulPartitionedCall"dropout_30/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?$
?
F__inference_initModel_layer_call_and_return_conditional_losses_7860973

inputs0
,dense_layer_1_matmul_readvariableop_resource1
-dense_layer_1_biasadd_readvariableop_resource0
,dense_layer_2_matmul_readvariableop_resource1
-dense_layer_2_biasadd_readvariableop_resource.
*logit_probs_matmul_readvariableop_resource/
+logit_probs_biasadd_readvariableop_resource
identity??
#Dense_Layer_1/MatMul/ReadVariableOpReadVariableOp,dense_layer_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#Dense_Layer_1/MatMul/ReadVariableOp?
Dense_Layer_1/MatMulMatMulinputs+Dense_Layer_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_Layer_1/MatMul?
$Dense_Layer_1/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$Dense_Layer_1/BiasAdd/ReadVariableOp?
Dense_Layer_1/BiasAddBiasAddDense_Layer_1/MatMul:product:0,Dense_Layer_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_Layer_1/BiasAdd?
Dense_Layer_1/ReluReluDense_Layer_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Dense_Layer_1/Reluy
dropout_30/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_30/dropout/Const?
dropout_30/dropout/MulMul Dense_Layer_1/Relu:activations:0!dropout_30/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_30/dropout/Mul?
dropout_30/dropout/ShapeShape Dense_Layer_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_30/dropout/Shape?
/dropout_30/dropout/random_uniform/RandomUniformRandomUniform!dropout_30/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*

seed21
/dropout_30/dropout/random_uniform/RandomUniform?
!dropout_30/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_30/dropout/GreaterEqual/y?
dropout_30/dropout/GreaterEqualGreaterEqual8dropout_30/dropout/random_uniform/RandomUniform:output:0*dropout_30/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
dropout_30/dropout/GreaterEqual?
dropout_30/dropout/CastCast#dropout_30/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_30/dropout/Cast?
dropout_30/dropout/Mul_1Muldropout_30/dropout/Mul:z:0dropout_30/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_30/dropout/Mul_1?
#Dense_Layer_2/MatMul/ReadVariableOpReadVariableOp,dense_layer_2_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02%
#Dense_Layer_2/MatMul/ReadVariableOp?
Dense_Layer_2/MatMulMatMuldropout_30/dropout/Mul_1:z:0+Dense_Layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
Dense_Layer_2/MatMul?
$Dense_Layer_2/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02&
$Dense_Layer_2/BiasAdd/ReadVariableOp?
Dense_Layer_2/BiasAddBiasAddDense_Layer_2/MatMul:product:0,Dense_Layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
Dense_Layer_2/BiasAdd?
Dense_Layer_2/ReluReluDense_Layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Dense_Layer_2/Relu?
!Logit_Probs/MatMul/ReadVariableOpReadVariableOp*logit_probs_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02#
!Logit_Probs/MatMul/ReadVariableOp?
Logit_Probs/MatMulMatMul Dense_Layer_2/Relu:activations:0)Logit_Probs/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Logit_Probs/MatMul?
"Logit_Probs/BiasAdd/ReadVariableOpReadVariableOp+logit_probs_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02$
"Logit_Probs/BiasAdd/ReadVariableOp?
Logit_Probs/BiasAddBiasAddLogit_Probs/MatMul:product:0*Logit_Probs/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Logit_Probs/BiasAddp
IdentityIdentityLogit_Probs/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????:::::::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
J__inference_Dense_Layer_1_layer_call_and_return_conditional_losses_7860719

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
"__inference__wrapped_model_7860704	
input:
6initmodel_dense_layer_1_matmul_readvariableop_resource;
7initmodel_dense_layer_1_biasadd_readvariableop_resource:
6initmodel_dense_layer_2_matmul_readvariableop_resource;
7initmodel_dense_layer_2_biasadd_readvariableop_resource8
4initmodel_logit_probs_matmul_readvariableop_resource9
5initmodel_logit_probs_biasadd_readvariableop_resource
identity??
-initModel/Dense_Layer_1/MatMul/ReadVariableOpReadVariableOp6initmodel_dense_layer_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-initModel/Dense_Layer_1/MatMul/ReadVariableOp?
initModel/Dense_Layer_1/MatMulMatMulinput5initModel/Dense_Layer_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
initModel/Dense_Layer_1/MatMul?
.initModel/Dense_Layer_1/BiasAdd/ReadVariableOpReadVariableOp7initmodel_dense_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.initModel/Dense_Layer_1/BiasAdd/ReadVariableOp?
initModel/Dense_Layer_1/BiasAddBiasAdd(initModel/Dense_Layer_1/MatMul:product:06initModel/Dense_Layer_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
initModel/Dense_Layer_1/BiasAdd?
initModel/Dense_Layer_1/ReluRelu(initModel/Dense_Layer_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
initModel/Dense_Layer_1/Relu?
initModel/dropout_30/IdentityIdentity*initModel/Dense_Layer_1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
initModel/dropout_30/Identity?
-initModel/Dense_Layer_2/MatMul/ReadVariableOpReadVariableOp6initmodel_dense_layer_2_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02/
-initModel/Dense_Layer_2/MatMul/ReadVariableOp?
initModel/Dense_Layer_2/MatMulMatMul&initModel/dropout_30/Identity:output:05initModel/Dense_Layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
initModel/Dense_Layer_2/MatMul?
.initModel/Dense_Layer_2/BiasAdd/ReadVariableOpReadVariableOp7initmodel_dense_layer_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype020
.initModel/Dense_Layer_2/BiasAdd/ReadVariableOp?
initModel/Dense_Layer_2/BiasAddBiasAdd(initModel/Dense_Layer_2/MatMul:product:06initModel/Dense_Layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2!
initModel/Dense_Layer_2/BiasAdd?
initModel/Dense_Layer_2/ReluRelu(initModel/Dense_Layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
initModel/Dense_Layer_2/Relu?
+initModel/Logit_Probs/MatMul/ReadVariableOpReadVariableOp4initmodel_logit_probs_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02-
+initModel/Logit_Probs/MatMul/ReadVariableOp?
initModel/Logit_Probs/MatMulMatMul*initModel/Dense_Layer_2/Relu:activations:03initModel/Logit_Probs/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
initModel/Logit_Probs/MatMul?
,initModel/Logit_Probs/BiasAdd/ReadVariableOpReadVariableOp5initmodel_logit_probs_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02.
,initModel/Logit_Probs/BiasAdd/ReadVariableOp?
initModel/Logit_Probs/BiasAddBiasAdd&initModel/Logit_Probs/MatMul:product:04initModel/Logit_Probs/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
initModel/Logit_Probs/BiasAddz
IdentityIdentity&initModel/Logit_Probs/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????:::::::O K
(
_output_shapes
:??????????

_user_specified_nameInput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
H
,__inference_dropout_30_layer_call_fn_7861079

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_30_layer_call_and_return_conditional_losses_78607522
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_initModel_layer_call_fn_7861032

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_initModel_layer_call_and_return_conditional_losses_78608992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
/__inference_Dense_Layer_2_layer_call_fn_7861099

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_Dense_Layer_2_layer_call_and_return_conditional_losses_78607762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_7860802

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d:::O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
e
,__inference_dropout_30_layer_call_fn_7861074

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_30_layer_call_and_return_conditional_losses_78607472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_initModel_layer_call_and_return_conditional_losses_7860899

inputs
dense_layer_1_7860882
dense_layer_1_7860884
dense_layer_2_7860888
dense_layer_2_7860890
logit_probs_7860893
logit_probs_7860895
identity??%Dense_Layer_1/StatefulPartitionedCall?%Dense_Layer_2/StatefulPartitionedCall?#Logit_Probs/StatefulPartitionedCall?
%Dense_Layer_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_layer_1_7860882dense_layer_1_7860884*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_Dense_Layer_1_layer_call_and_return_conditional_losses_78607192'
%Dense_Layer_1/StatefulPartitionedCall?
dropout_30/PartitionedCallPartitionedCall.Dense_Layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_30_layer_call_and_return_conditional_losses_78607522
dropout_30/PartitionedCall?
%Dense_Layer_2/StatefulPartitionedCallStatefulPartitionedCall#dropout_30/PartitionedCall:output:0dense_layer_2_7860888dense_layer_2_7860890*
Tin
2*
Tout
2*'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_Dense_Layer_2_layer_call_and_return_conditional_losses_78607762'
%Dense_Layer_2/StatefulPartitionedCall?
#Logit_Probs/StatefulPartitionedCallStatefulPartitionedCall.Dense_Layer_2/StatefulPartitionedCall:output:0logit_probs_7860893logit_probs_7860895*
Tin
2*
Tout
2*'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*Q
fLRJ
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_78608022%
#Logit_Probs/StatefulPartitionedCall?
IdentityIdentity,Logit_Probs/StatefulPartitionedCall:output:0&^Dense_Layer_1/StatefulPartitionedCall&^Dense_Layer_2/StatefulPartitionedCall$^Logit_Probs/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::2N
%Dense_Layer_1/StatefulPartitionedCall%Dense_Layer_1/StatefulPartitionedCall2N
%Dense_Layer_2/StatefulPartitionedCall%Dense_Layer_2/StatefulPartitionedCall2J
#Logit_Probs/StatefulPartitionedCall#Logit_Probs/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
F__inference_initModel_layer_call_and_return_conditional_losses_7860819	
input
dense_layer_1_7860730
dense_layer_1_7860732
dense_layer_2_7860787
dense_layer_2_7860789
logit_probs_7860813
logit_probs_7860815
identity??%Dense_Layer_1/StatefulPartitionedCall?%Dense_Layer_2/StatefulPartitionedCall?#Logit_Probs/StatefulPartitionedCall?"dropout_30/StatefulPartitionedCall?
%Dense_Layer_1/StatefulPartitionedCallStatefulPartitionedCallinputdense_layer_1_7860730dense_layer_1_7860732*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_Dense_Layer_1_layer_call_and_return_conditional_losses_78607192'
%Dense_Layer_1/StatefulPartitionedCall?
"dropout_30/StatefulPartitionedCallStatefulPartitionedCall.Dense_Layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_30_layer_call_and_return_conditional_losses_78607472$
"dropout_30/StatefulPartitionedCall?
%Dense_Layer_2/StatefulPartitionedCallStatefulPartitionedCall+dropout_30/StatefulPartitionedCall:output:0dense_layer_2_7860787dense_layer_2_7860789*
Tin
2*
Tout
2*'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_Dense_Layer_2_layer_call_and_return_conditional_losses_78607762'
%Dense_Layer_2/StatefulPartitionedCall?
#Logit_Probs/StatefulPartitionedCallStatefulPartitionedCall.Dense_Layer_2/StatefulPartitionedCall:output:0logit_probs_7860813logit_probs_7860815*
Tin
2*
Tout
2*'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*Q
fLRJ
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_78608022%
#Logit_Probs/StatefulPartitionedCall?
IdentityIdentity,Logit_Probs/StatefulPartitionedCall:output:0&^Dense_Layer_1/StatefulPartitionedCall&^Dense_Layer_2/StatefulPartitionedCall$^Logit_Probs/StatefulPartitionedCall#^dropout_30/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::2N
%Dense_Layer_1/StatefulPartitionedCall%Dense_Layer_1/StatefulPartitionedCall2N
%Dense_Layer_2/StatefulPartitionedCall%Dense_Layer_2/StatefulPartitionedCall2J
#Logit_Probs/StatefulPartitionedCall#Logit_Probs/StatefulPartitionedCall2H
"dropout_30/StatefulPartitionedCall"dropout_30/StatefulPartitionedCall:O K
(
_output_shapes
:??????????

_user_specified_nameInput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
e
G__inference_dropout_30_layer_call_and_return_conditional_losses_7860752

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_Logit_Probs_layer_call_fn_7861118

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*Q
fLRJ
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_78608022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
8
Input/
serving_default_Input:0???????????
Logit_Probs0
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:̤
?*
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
	optimizer
regularization_losses
trainable_variables
		variables

	keras_api

signatures
]_default_save_signature
*^&call_and_return_all_conditional_losses
___call__"?(
_tf_keras_model?'{"class_name": "Model", "name": "initModel", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "initModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}, "name": "Input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "Dense_Layer_1", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_Layer_1", "inbound_nodes": [[["Input", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_30", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_30", "inbound_nodes": [[["Dense_Layer_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_Layer_2", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_Layer_2", "inbound_nodes": [[["dropout_30", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Logit_Probs", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Logit_Probs", "inbound_nodes": [[["Dense_Layer_2", 0, 0, {}]]]}], "input_layers": [["Input", 0, 0]], "output_layers": [["Logit_Probs", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "initModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}, "name": "Input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "Dense_Layer_1", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_Layer_1", "inbound_nodes": [[["Input", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_30", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_30", "inbound_nodes": [[["Dense_Layer_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_Layer_2", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_Layer_2", "inbound_nodes": [[["dropout_30", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Logit_Probs", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Logit_Probs", "inbound_nodes": [[["Dense_Layer_2", 0, 0, {}]]]}], "input_layers": [["Input", 0, 0]], "output_layers": [["Logit_Probs", 0, 0]]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}}, "metrics": ["accuracy", {"class_name": "SparseTopKCategoricalAccuracy", "config": {"name": "sparse_top_k_categorical_accuracy", "dtype": "float32", "k": 3}}], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "Input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*`&call_and_return_all_conditional_losses
a__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "Dense_Layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Dense_Layer_1", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
?
regularization_losses
trainable_variables
	variables
	keras_api
*b&call_and_return_all_conditional_losses
c__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_30", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_30", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*d&call_and_return_all_conditional_losses
e__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "Dense_Layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Dense_Layer_2", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
?

kernel
bias
regularization_losses
trainable_variables
 	variables
!	keras_api
*f&call_and_return_all_conditional_losses
g__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "Logit_Probs", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Logit_Probs", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?
"iter

#beta_1

$beta_2
	%decay
&learning_ratemQmRmSmTmUmVvWvXvYvZv[v\"
	optimizer
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
?
regularization_losses

'layers
trainable_variables
		variables
(layer_regularization_losses
)layer_metrics
*metrics
+non_trainable_variables
___call__
]_default_save_signature
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
,
hserving_default"
signature_map
+:)
??2Dense_Layer_1_30/kernel
$:"?2Dense_Layer_1_30/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses

,layers
-layer_regularization_losses
trainable_variables
	variables
.layer_metrics
/metrics
0non_trainable_variables
a__call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses

1layers
2layer_regularization_losses
trainable_variables
	variables
3layer_metrics
4metrics
5non_trainable_variables
c__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
*:(	?d2Dense_Layer_2_19/kernel
#:!d2Dense_Layer_2_19/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses

6layers
7layer_regularization_losses
trainable_variables
	variables
8layer_metrics
9metrics
:non_trainable_variables
e__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
':%d
2Logit_Probs_30/kernel
!:
2Logit_Probs_30/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses

;layers
<layer_regularization_losses
trainable_variables
 	variables
=layer_metrics
>metrics
?non_trainable_variables
g__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
@0
A1
B2"
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
	Ctotal
	Dcount
E	variables
F	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	Gtotal
	Hcount
I
_fn_kwargs
J	variables
K	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
?
	Ltotal
	Mcount
N
_fn_kwargs
O	variables
P	keras_api"?
_tf_keras_metric?{"class_name": "SparseTopKCategoricalAccuracy", "name": "sparse_top_k_categorical_accuracy", "dtype": "float32", "config": {"name": "sparse_top_k_categorical_accuracy", "dtype": "float32", "k": 3}}
:  (2total
:  (2count
.
C0
D1"
trackable_list_wrapper
-
E	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
G0
H1"
trackable_list_wrapper
-
J	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
L0
M1"
trackable_list_wrapper
-
O	variables"
_generic_user_object
0:.
??2Adam/Dense_Layer_1_30/kernel/m
):'?2Adam/Dense_Layer_1_30/bias/m
/:-	?d2Adam/Dense_Layer_2_19/kernel/m
(:&d2Adam/Dense_Layer_2_19/bias/m
,:*d
2Adam/Logit_Probs_30/kernel/m
&:$
2Adam/Logit_Probs_30/bias/m
0:.
??2Adam/Dense_Layer_1_30/kernel/v
):'?2Adam/Dense_Layer_1_30/bias/v
/:-	?d2Adam/Dense_Layer_2_19/kernel/v
(:&d2Adam/Dense_Layer_2_19/bias/v
,:*d
2Adam/Logit_Probs_30/kernel/v
&:$
2Adam/Logit_Probs_30/bias/v
?2?
"__inference__wrapped_model_7860704?
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
annotations? *%?"
 ?
Input??????????
?2?
F__inference_initModel_layer_call_and_return_conditional_losses_7860973
F__inference_initModel_layer_call_and_return_conditional_losses_7860998
F__inference_initModel_layer_call_and_return_conditional_losses_7860839
F__inference_initModel_layer_call_and_return_conditional_losses_7860819?
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
+__inference_initModel_layer_call_fn_7861032
+__inference_initModel_layer_call_fn_7860914
+__inference_initModel_layer_call_fn_7860877
+__inference_initModel_layer_call_fn_7861015?
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
J__inference_Dense_Layer_1_layer_call_and_return_conditional_losses_7861043?
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
/__inference_Dense_Layer_1_layer_call_fn_7861052?
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
?2?
G__inference_dropout_30_layer_call_and_return_conditional_losses_7861069
G__inference_dropout_30_layer_call_and_return_conditional_losses_7861064?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_dropout_30_layer_call_fn_7861079
,__inference_dropout_30_layer_call_fn_7861074?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_Dense_Layer_2_layer_call_and_return_conditional_losses_7861090?
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
/__inference_Dense_Layer_2_layer_call_fn_7861099?
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
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_7861109?
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
-__inference_Logit_Probs_layer_call_fn_7861118?
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
2B0
%__inference_signature_wrapper_7860941Input?
J__inference_Dense_Layer_1_layer_call_and_return_conditional_losses_7861043^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
/__inference_Dense_Layer_1_layer_call_fn_7861052Q0?-
&?#
!?
inputs??????????
? "????????????
J__inference_Dense_Layer_2_layer_call_and_return_conditional_losses_7861090]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????d
? ?
/__inference_Dense_Layer_2_layer_call_fn_7861099P0?-
&?#
!?
inputs??????????
? "??????????d?
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_7861109\/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????

? ?
-__inference_Logit_Probs_layer_call_fn_7861118O/?,
%?"
 ?
inputs?????????d
? "??????????
?
"__inference__wrapped_model_7860704t/?,
%?"
 ?
Input??????????
? "9?6
4
Logit_Probs%?"
Logit_Probs?????????
?
G__inference_dropout_30_layer_call_and_return_conditional_losses_7861064^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
G__inference_dropout_30_layer_call_and_return_conditional_losses_7861069^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
,__inference_dropout_30_layer_call_fn_7861074Q4?1
*?'
!?
inputs??????????
p
? "????????????
,__inference_dropout_30_layer_call_fn_7861079Q4?1
*?'
!?
inputs??????????
p 
? "????????????
F__inference_initModel_layer_call_and_return_conditional_losses_7860819h7?4
-?*
 ?
Input??????????
p

 
? "%?"
?
0?????????

? ?
F__inference_initModel_layer_call_and_return_conditional_losses_7860839h7?4
-?*
 ?
Input??????????
p 

 
? "%?"
?
0?????????

? ?
F__inference_initModel_layer_call_and_return_conditional_losses_7860973i8?5
.?+
!?
inputs??????????
p

 
? "%?"
?
0?????????

? ?
F__inference_initModel_layer_call_and_return_conditional_losses_7860998i8?5
.?+
!?
inputs??????????
p 

 
? "%?"
?
0?????????

? ?
+__inference_initModel_layer_call_fn_7860877[7?4
-?*
 ?
Input??????????
p

 
? "??????????
?
+__inference_initModel_layer_call_fn_7860914[7?4
-?*
 ?
Input??????????
p 

 
? "??????????
?
+__inference_initModel_layer_call_fn_7861015\8?5
.?+
!?
inputs??????????
p

 
? "??????????
?
+__inference_initModel_layer_call_fn_7861032\8?5
.?+
!?
inputs??????????
p 

 
? "??????????
?
%__inference_signature_wrapper_7860941}8?5
? 
.?+
)
Input ?
Input??????????"9?6
4
Logit_Probs%?"
Logit_Probs?????????
