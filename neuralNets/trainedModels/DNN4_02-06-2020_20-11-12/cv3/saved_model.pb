??
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
shapeshape?"serve*2.2.0-dlenv2v2.2.0-0-g2b96f368ۉ	
?
Dense_layer_1_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameDense_layer_1_36/kernel
?
+Dense_layer_1_36/kernel/Read/ReadVariableOpReadVariableOpDense_layer_1_36/kernel* 
_output_shapes
:
??*
dtype0
?
Dense_layer_1_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameDense_layer_1_36/bias
|
)Dense_layer_1_36/bias/Read/ReadVariableOpReadVariableOpDense_layer_1_36/bias*
_output_shapes	
:?*
dtype0
?
Dense_layer_2_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*(
shared_nameDense_layer_2_25/kernel
?
+Dense_layer_2_25/kernel/Read/ReadVariableOpReadVariableOpDense_layer_2_25/kernel*
_output_shapes
:	?d*
dtype0
?
Dense_layer_2_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameDense_layer_2_25/bias
{
)Dense_layer_2_25/bias/Read/ReadVariableOpReadVariableOpDense_layer_2_25/bias*
_output_shapes
:d*
dtype0
?
Dense_layer_3_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameDense_layer_3_3/kernel
?
*Dense_layer_3_3/kernel/Read/ReadVariableOpReadVariableOpDense_layer_3_3/kernel*
_output_shapes

:dd*
dtype0
?
Dense_layer_3_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameDense_layer_3_3/bias
y
(Dense_layer_3_3/bias/Read/ReadVariableOpReadVariableOpDense_layer_3_3/bias*
_output_shapes
:d*
dtype0
?
Logit_Probs_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*&
shared_nameLogit_Probs_36/kernel

)Logit_Probs_36/kernel/Read/ReadVariableOpReadVariableOpLogit_Probs_36/kernel*
_output_shapes

:d
*
dtype0
~
Logit_Probs_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameLogit_Probs_36/bias
w
'Logit_Probs_36/bias/Read/ReadVariableOpReadVariableOpLogit_Probs_36/bias*
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
Adam/Dense_layer_1_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*/
shared_name Adam/Dense_layer_1_36/kernel/m
?
2Adam/Dense_layer_1_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_1_36/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/Dense_layer_1_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_nameAdam/Dense_layer_1_36/bias/m
?
0Adam/Dense_layer_1_36/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_1_36/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/Dense_layer_2_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*/
shared_name Adam/Dense_layer_2_25/kernel/m
?
2Adam/Dense_layer_2_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_2_25/kernel/m*
_output_shapes
:	?d*
dtype0
?
Adam/Dense_layer_2_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*-
shared_nameAdam/Dense_layer_2_25/bias/m
?
0Adam/Dense_layer_2_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_2_25/bias/m*
_output_shapes
:d*
dtype0
?
Adam/Dense_layer_3_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*.
shared_nameAdam/Dense_layer_3_3/kernel/m
?
1Adam/Dense_layer_3_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_3_3/kernel/m*
_output_shapes

:dd*
dtype0
?
Adam/Dense_layer_3_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*,
shared_nameAdam/Dense_layer_3_3/bias/m
?
/Adam/Dense_layer_3_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_3_3/bias/m*
_output_shapes
:d*
dtype0
?
Adam/Logit_Probs_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*-
shared_nameAdam/Logit_Probs_36/kernel/m
?
0Adam/Logit_Probs_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Logit_Probs_36/kernel/m*
_output_shapes

:d
*
dtype0
?
Adam/Logit_Probs_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_nameAdam/Logit_Probs_36/bias/m
?
.Adam/Logit_Probs_36/bias/m/Read/ReadVariableOpReadVariableOpAdam/Logit_Probs_36/bias/m*
_output_shapes
:
*
dtype0
?
Adam/Dense_layer_1_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*/
shared_name Adam/Dense_layer_1_36/kernel/v
?
2Adam/Dense_layer_1_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_1_36/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/Dense_layer_1_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_nameAdam/Dense_layer_1_36/bias/v
?
0Adam/Dense_layer_1_36/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_1_36/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/Dense_layer_2_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*/
shared_name Adam/Dense_layer_2_25/kernel/v
?
2Adam/Dense_layer_2_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_2_25/kernel/v*
_output_shapes
:	?d*
dtype0
?
Adam/Dense_layer_2_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*-
shared_nameAdam/Dense_layer_2_25/bias/v
?
0Adam/Dense_layer_2_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_2_25/bias/v*
_output_shapes
:d*
dtype0
?
Adam/Dense_layer_3_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*.
shared_nameAdam/Dense_layer_3_3/kernel/v
?
1Adam/Dense_layer_3_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_3_3/kernel/v*
_output_shapes

:dd*
dtype0
?
Adam/Dense_layer_3_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*,
shared_nameAdam/Dense_layer_3_3/bias/v
?
/Adam/Dense_layer_3_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_3_3/bias/v*
_output_shapes
:d*
dtype0
?
Adam/Logit_Probs_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*-
shared_nameAdam/Logit_Probs_36/kernel/v
?
0Adam/Logit_Probs_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Logit_Probs_36/kernel/v*
_output_shapes

:d
*
dtype0
?
Adam/Logit_Probs_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_nameAdam/Logit_Probs_36/bias/v
?
.Adam/Logit_Probs_36/bias/v/Read/ReadVariableOpReadVariableOpAdam/Logit_Probs_36/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
?;
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?:
value?:B?: B?:
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
R
$regularization_losses
%trainable_variables
&	variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
R
.regularization_losses
/trainable_variables
0	variables
1	keras_api
h

2kernel
3bias
4regularization_losses
5trainable_variables
6	variables
7	keras_api
?
8iter

9beta_1

:beta_2
	;decay
<learning_ratem{m|m}m~(m)m?2m?3m?v?v?v?v?(v?)v?2v?3v?
 
8
0
1
2
3
(4
)5
26
37
8
0
1
2
3
(4
)5
26
37
?
regularization_losses

=layers
trainable_variables
	variables
>layer_regularization_losses
?layer_metrics
@metrics
Anon_trainable_variables
 
 
 
 
?
regularization_losses

Blayers
Clayer_regularization_losses
trainable_variables
	variables
Dlayer_metrics
Emetrics
Fnon_trainable_variables
ca
VARIABLE_VALUEDense_layer_1_36/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEDense_layer_1_36/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses

Glayers
Hlayer_regularization_losses
trainable_variables
	variables
Ilayer_metrics
Jmetrics
Knon_trainable_variables
 
 
 
?
regularization_losses

Llayers
Mlayer_regularization_losses
trainable_variables
	variables
Nlayer_metrics
Ometrics
Pnon_trainable_variables
ca
VARIABLE_VALUEDense_layer_2_25/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEDense_layer_2_25/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
 regularization_losses

Qlayers
Rlayer_regularization_losses
!trainable_variables
"	variables
Slayer_metrics
Tmetrics
Unon_trainable_variables
 
 
 
?
$regularization_losses

Vlayers
Wlayer_regularization_losses
%trainable_variables
&	variables
Xlayer_metrics
Ymetrics
Znon_trainable_variables
b`
VARIABLE_VALUEDense_layer_3_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEDense_layer_3_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
?
*regularization_losses

[layers
\layer_regularization_losses
+trainable_variables
,	variables
]layer_metrics
^metrics
_non_trainable_variables
 
 
 
?
.regularization_losses

`layers
alayer_regularization_losses
/trainable_variables
0	variables
blayer_metrics
cmetrics
dnon_trainable_variables
a_
VARIABLE_VALUELogit_Probs_36/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUELogit_Probs_36/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

20
31

20
31
?
4regularization_losses

elayers
flayer_regularization_losses
5trainable_variables
6	variables
glayer_metrics
hmetrics
inon_trainable_variables
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
?
0
1
2
3
4
5
6
7
	8
 
 

j0
k1
l2
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
 
 
 
 
4
	mtotal
	ncount
o	variables
p	keras_api
D
	qtotal
	rcount
s
_fn_kwargs
t	variables
u	keras_api
D
	vtotal
	wcount
x
_fn_kwargs
y	variables
z	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

m0
n1

o	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

q0
r1

t	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

v0
w1

y	variables
??
VARIABLE_VALUEAdam/Dense_layer_1_36/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Dense_layer_1_36/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Dense_layer_2_25/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Dense_layer_2_25/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Dense_layer_3_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/Dense_layer_3_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Logit_Probs_36/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/Logit_Probs_36/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Dense_layer_1_36/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Dense_layer_1_36/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Dense_layer_2_25/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Dense_layer_2_25/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Dense_layer_3_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/Dense_layer_3_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Logit_Probs_36/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/Logit_Probs_36/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_InputPlaceholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_InputDense_layer_1_36/kernelDense_layer_1_36/biasDense_layer_2_25/kernelDense_layer_2_25/biasDense_layer_3_3/kernelDense_layer_3_3/biasLogit_Probs_36/kernelLogit_Probs_36/bias*
Tin
2	*
Tout
2*'
_output_shapes
:?????????
**
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*.
f)R'
%__inference_signature_wrapper_9445523
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+Dense_layer_1_36/kernel/Read/ReadVariableOp)Dense_layer_1_36/bias/Read/ReadVariableOp+Dense_layer_2_25/kernel/Read/ReadVariableOp)Dense_layer_2_25/bias/Read/ReadVariableOp*Dense_layer_3_3/kernel/Read/ReadVariableOp(Dense_layer_3_3/bias/Read/ReadVariableOp)Logit_Probs_36/kernel/Read/ReadVariableOp'Logit_Probs_36/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp2Adam/Dense_layer_1_36/kernel/m/Read/ReadVariableOp0Adam/Dense_layer_1_36/bias/m/Read/ReadVariableOp2Adam/Dense_layer_2_25/kernel/m/Read/ReadVariableOp0Adam/Dense_layer_2_25/bias/m/Read/ReadVariableOp1Adam/Dense_layer_3_3/kernel/m/Read/ReadVariableOp/Adam/Dense_layer_3_3/bias/m/Read/ReadVariableOp0Adam/Logit_Probs_36/kernel/m/Read/ReadVariableOp.Adam/Logit_Probs_36/bias/m/Read/ReadVariableOp2Adam/Dense_layer_1_36/kernel/v/Read/ReadVariableOp0Adam/Dense_layer_1_36/bias/v/Read/ReadVariableOp2Adam/Dense_layer_2_25/kernel/v/Read/ReadVariableOp0Adam/Dense_layer_2_25/bias/v/Read/ReadVariableOp1Adam/Dense_layer_3_3/kernel/v/Read/ReadVariableOp/Adam/Dense_layer_3_3/bias/v/Read/ReadVariableOp0Adam/Logit_Probs_36/kernel/v/Read/ReadVariableOp.Adam/Logit_Probs_36/bias/v/Read/ReadVariableOpConst*0
Tin)
'2%	*
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
 __inference__traced_save_9446045
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameDense_layer_1_36/kernelDense_layer_1_36/biasDense_layer_2_25/kernelDense_layer_2_25/biasDense_layer_3_3/kernelDense_layer_3_3/biasLogit_Probs_36/kernelLogit_Probs_36/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/Dense_layer_1_36/kernel/mAdam/Dense_layer_1_36/bias/mAdam/Dense_layer_2_25/kernel/mAdam/Dense_layer_2_25/bias/mAdam/Dense_layer_3_3/kernel/mAdam/Dense_layer_3_3/bias/mAdam/Logit_Probs_36/kernel/mAdam/Logit_Probs_36/bias/mAdam/Dense_layer_1_36/kernel/vAdam/Dense_layer_1_36/bias/vAdam/Dense_layer_2_25/kernel/vAdam/Dense_layer_2_25/bias/vAdam/Dense_layer_3_3/kernel/vAdam/Dense_layer_3_3/bias/vAdam/Logit_Probs_36/kernel/vAdam/Logit_Probs_36/bias/v*/
Tin(
&2$*
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
#__inference__traced_restore_9446162??
?
?
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_9445881

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
?[
?
F__inference_initModel_layer_call_and_return_conditional_losses_9445596

inputs0
,dense_layer_1_matmul_readvariableop_resource1
-dense_layer_1_biasadd_readvariableop_resource0
,dense_layer_2_matmul_readvariableop_resource1
-dense_layer_2_biasadd_readvariableop_resource0
,dense_layer_3_matmul_readvariableop_resource1
-dense_layer_3_biasadd_readvariableop_resource.
*logit_probs_matmul_readvariableop_resource/
+logit_probs_biasadd_readvariableop_resource
identity?y
dropout_45/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_45/dropout/Const?
dropout_45/dropout/MulMulinputs!dropout_45/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_45/dropout/Mulj
dropout_45/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout_45/dropout/Shape?
/dropout_45/dropout/random_uniform/RandomUniformRandomUniform!dropout_45/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*

seed21
/dropout_45/dropout/random_uniform/RandomUniform?
!dropout_45/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_45/dropout/GreaterEqual/y?
dropout_45/dropout/GreaterEqualGreaterEqual8dropout_45/dropout/random_uniform/RandomUniform:output:0*dropout_45/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
dropout_45/dropout/GreaterEqual?
dropout_45/dropout/CastCast#dropout_45/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_45/dropout/Cast?
dropout_45/dropout/Mul_1Muldropout_45/dropout/Mul:z:0dropout_45/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_45/dropout/Mul_1?
#Dense_layer_1/MatMul/ReadVariableOpReadVariableOp,dense_layer_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#Dense_layer_1/MatMul/ReadVariableOp?
Dense_layer_1/MatMulMatMuldropout_45/dropout/Mul_1:z:0+Dense_layer_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_layer_1/MatMul?
$Dense_layer_1/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$Dense_layer_1/BiasAdd/ReadVariableOp?
Dense_layer_1/BiasAddBiasAddDense_layer_1/MatMul:product:0,Dense_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_layer_1/BiasAdd?
Dense_layer_1/TanhTanhDense_layer_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Dense_layer_1/Tanhy
dropout_46/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_46/dropout/Const?
dropout_46/dropout/MulMulDense_layer_1/Tanh:y:0!dropout_46/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_46/dropout/Mulz
dropout_46/dropout/ShapeShapeDense_layer_1/Tanh:y:0*
T0*
_output_shapes
:2
dropout_46/dropout/Shape?
/dropout_46/dropout/random_uniform/RandomUniformRandomUniform!dropout_46/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*

seed*
seed221
/dropout_46/dropout/random_uniform/RandomUniform?
!dropout_46/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dropout_46/dropout/GreaterEqual/y?
dropout_46/dropout/GreaterEqualGreaterEqual8dropout_46/dropout/random_uniform/RandomUniform:output:0*dropout_46/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
dropout_46/dropout/GreaterEqual?
dropout_46/dropout/CastCast#dropout_46/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_46/dropout/Cast?
dropout_46/dropout/Mul_1Muldropout_46/dropout/Mul:z:0dropout_46/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_46/dropout/Mul_1?
#Dense_layer_2/MatMul/ReadVariableOpReadVariableOp,dense_layer_2_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02%
#Dense_layer_2/MatMul/ReadVariableOp?
Dense_layer_2/MatMulMatMuldropout_46/dropout/Mul_1:z:0+Dense_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
Dense_layer_2/MatMul?
$Dense_layer_2/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02&
$Dense_layer_2/BiasAdd/ReadVariableOp?
Dense_layer_2/BiasAddBiasAddDense_layer_2/MatMul:product:0,Dense_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
Dense_layer_2/BiasAdd?
Dense_layer_2/TanhTanhDense_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Dense_layer_2/Tanhy
dropout_47/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_47/dropout/Const?
dropout_47/dropout/MulMulDense_layer_2/Tanh:y:0!dropout_47/dropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout_47/dropout/Mulz
dropout_47/dropout/ShapeShapeDense_layer_2/Tanh:y:0*
T0*
_output_shapes
:2
dropout_47/dropout/Shape?
/dropout_47/dropout/random_uniform/RandomUniformRandomUniform!dropout_47/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0*

seed*
seed221
/dropout_47/dropout/random_uniform/RandomUniform?
!dropout_47/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dropout_47/dropout/GreaterEqual/y?
dropout_47/dropout/GreaterEqualGreaterEqual8dropout_47/dropout/random_uniform/RandomUniform:output:0*dropout_47/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2!
dropout_47/dropout/GreaterEqual?
dropout_47/dropout/CastCast#dropout_47/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout_47/dropout/Cast?
dropout_47/dropout/Mul_1Muldropout_47/dropout/Mul:z:0dropout_47/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout_47/dropout/Mul_1?
#Dense_layer_3/MatMul/ReadVariableOpReadVariableOp,dense_layer_3_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02%
#Dense_layer_3/MatMul/ReadVariableOp?
Dense_layer_3/MatMulMatMuldropout_47/dropout/Mul_1:z:0+Dense_layer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
Dense_layer_3/MatMul?
$Dense_layer_3/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02&
$Dense_layer_3/BiasAdd/ReadVariableOp?
Dense_layer_3/BiasAddBiasAddDense_layer_3/MatMul:product:0,Dense_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
Dense_layer_3/BiasAdd?
Dense_layer_3/TanhTanhDense_layer_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Dense_layer_3/Tanhy
dropout_48/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_48/dropout/Const?
dropout_48/dropout/MulMulDense_layer_3/Tanh:y:0!dropout_48/dropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout_48/dropout/Mulz
dropout_48/dropout/ShapeShapeDense_layer_3/Tanh:y:0*
T0*
_output_shapes
:2
dropout_48/dropout/Shape?
/dropout_48/dropout/random_uniform/RandomUniformRandomUniform!dropout_48/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0*

seed*
seed221
/dropout_48/dropout/random_uniform/RandomUniform?
!dropout_48/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dropout_48/dropout/GreaterEqual/y?
dropout_48/dropout/GreaterEqualGreaterEqual8dropout_48/dropout/random_uniform/RandomUniform:output:0*dropout_48/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2!
dropout_48/dropout/GreaterEqual?
dropout_48/dropout/CastCast#dropout_48/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout_48/dropout/Cast?
dropout_48/dropout/Mul_1Muldropout_48/dropout/Mul:z:0dropout_48/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout_48/dropout/Mul_1?
!Logit_Probs/MatMul/ReadVariableOpReadVariableOp*logit_probs_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02#
!Logit_Probs/MatMul/ReadVariableOp?
Logit_Probs/MatMulMatMuldropout_48/dropout/Mul_1:z:0)Logit_Probs/MatMul/ReadVariableOp:value:0*
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
Logit_Probs/BiasAdd?
6Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp,dense_layer_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype028
6Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOp?
'Dense_layer_1_36/kernel/Regularizer/AbsAbs>Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2)
'Dense_layer_1_36/kernel/Regularizer/Abs?
)Dense_layer_1_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)Dense_layer_1_36/kernel/Regularizer/Const?
'Dense_layer_1_36/kernel/Regularizer/SumSum+Dense_layer_1_36/kernel/Regularizer/Abs:y:02Dense_layer_1_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/Sum?
)Dense_layer_1_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2+
)Dense_layer_1_36/kernel/Regularizer/mul/x?
'Dense_layer_1_36/kernel/Regularizer/mulMul2Dense_layer_1_36/kernel/Regularizer/mul/x:output:00Dense_layer_1_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/mul?
)Dense_layer_1_36/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_1_36/kernel/Regularizer/add/x?
'Dense_layer_1_36/kernel/Regularizer/addAddV22Dense_layer_1_36/kernel/Regularizer/add/x:output:0+Dense_layer_1_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/add?
)Dense_layer_2_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_2_25/kernel/Regularizer/Const?
(Dense_layer_3_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(Dense_layer_3_3/kernel/Regularizer/Constp
IdentityIdentityLogit_Probs/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????:::::::::P L
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
: :

_output_shapes
: :

_output_shapes
: 
?,
?
"__inference__wrapped_model_9445073	
input:
6initmodel_dense_layer_1_matmul_readvariableop_resource;
7initmodel_dense_layer_1_biasadd_readvariableop_resource:
6initmodel_dense_layer_2_matmul_readvariableop_resource;
7initmodel_dense_layer_2_biasadd_readvariableop_resource:
6initmodel_dense_layer_3_matmul_readvariableop_resource;
7initmodel_dense_layer_3_biasadd_readvariableop_resource8
4initmodel_logit_probs_matmul_readvariableop_resource9
5initmodel_logit_probs_biasadd_readvariableop_resource
identity??
initModel/dropout_45/IdentityIdentityinput*
T0*(
_output_shapes
:??????????2
initModel/dropout_45/Identity?
-initModel/Dense_layer_1/MatMul/ReadVariableOpReadVariableOp6initmodel_dense_layer_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-initModel/Dense_layer_1/MatMul/ReadVariableOp?
initModel/Dense_layer_1/MatMulMatMul&initModel/dropout_45/Identity:output:05initModel/Dense_layer_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
initModel/Dense_layer_1/MatMul?
.initModel/Dense_layer_1/BiasAdd/ReadVariableOpReadVariableOp7initmodel_dense_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.initModel/Dense_layer_1/BiasAdd/ReadVariableOp?
initModel/Dense_layer_1/BiasAddBiasAdd(initModel/Dense_layer_1/MatMul:product:06initModel/Dense_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
initModel/Dense_layer_1/BiasAdd?
initModel/Dense_layer_1/TanhTanh(initModel/Dense_layer_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
initModel/Dense_layer_1/Tanh?
initModel/dropout_46/IdentityIdentity initModel/Dense_layer_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
initModel/dropout_46/Identity?
-initModel/Dense_layer_2/MatMul/ReadVariableOpReadVariableOp6initmodel_dense_layer_2_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02/
-initModel/Dense_layer_2/MatMul/ReadVariableOp?
initModel/Dense_layer_2/MatMulMatMul&initModel/dropout_46/Identity:output:05initModel/Dense_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
initModel/Dense_layer_2/MatMul?
.initModel/Dense_layer_2/BiasAdd/ReadVariableOpReadVariableOp7initmodel_dense_layer_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype020
.initModel/Dense_layer_2/BiasAdd/ReadVariableOp?
initModel/Dense_layer_2/BiasAddBiasAdd(initModel/Dense_layer_2/MatMul:product:06initModel/Dense_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2!
initModel/Dense_layer_2/BiasAdd?
initModel/Dense_layer_2/TanhTanh(initModel/Dense_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
initModel/Dense_layer_2/Tanh?
initModel/dropout_47/IdentityIdentity initModel/Dense_layer_2/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
initModel/dropout_47/Identity?
-initModel/Dense_layer_3/MatMul/ReadVariableOpReadVariableOp6initmodel_dense_layer_3_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02/
-initModel/Dense_layer_3/MatMul/ReadVariableOp?
initModel/Dense_layer_3/MatMulMatMul&initModel/dropout_47/Identity:output:05initModel/Dense_layer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
initModel/Dense_layer_3/MatMul?
.initModel/Dense_layer_3/BiasAdd/ReadVariableOpReadVariableOp7initmodel_dense_layer_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype020
.initModel/Dense_layer_3/BiasAdd/ReadVariableOp?
initModel/Dense_layer_3/BiasAddBiasAdd(initModel/Dense_layer_3/MatMul:product:06initModel/Dense_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2!
initModel/Dense_layer_3/BiasAdd?
initModel/Dense_layer_3/TanhTanh(initModel/Dense_layer_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
initModel/Dense_layer_3/Tanh?
initModel/dropout_48/IdentityIdentity initModel/Dense_layer_3/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
initModel/dropout_48/Identity?
+initModel/Logit_Probs/MatMul/ReadVariableOpReadVariableOp4initmodel_logit_probs_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02-
+initModel/Logit_Probs/MatMul/ReadVariableOp?
initModel/Logit_Probs/MatMulMatMul&initModel/dropout_48/Identity:output:03initModel/Logit_Probs/MatMul/ReadVariableOp:value:0*
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
identityIdentity:output:0*G
_input_shapes6
4:??????????:::::::::O K
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
: :

_output_shapes
: :

_output_shapes
: 
?
f
G__inference_dropout_45_layer_call_and_return_conditional_losses_9445089

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?T
?
 __inference__traced_save_9446045
file_prefix6
2savev2_dense_layer_1_36_kernel_read_readvariableop4
0savev2_dense_layer_1_36_bias_read_readvariableop6
2savev2_dense_layer_2_25_kernel_read_readvariableop4
0savev2_dense_layer_2_25_bias_read_readvariableop5
1savev2_dense_layer_3_3_kernel_read_readvariableop3
/savev2_dense_layer_3_3_bias_read_readvariableop4
0savev2_logit_probs_36_kernel_read_readvariableop2
.savev2_logit_probs_36_bias_read_readvariableop(
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
9savev2_adam_dense_layer_1_36_kernel_m_read_readvariableop;
7savev2_adam_dense_layer_1_36_bias_m_read_readvariableop=
9savev2_adam_dense_layer_2_25_kernel_m_read_readvariableop;
7savev2_adam_dense_layer_2_25_bias_m_read_readvariableop<
8savev2_adam_dense_layer_3_3_kernel_m_read_readvariableop:
6savev2_adam_dense_layer_3_3_bias_m_read_readvariableop;
7savev2_adam_logit_probs_36_kernel_m_read_readvariableop9
5savev2_adam_logit_probs_36_bias_m_read_readvariableop=
9savev2_adam_dense_layer_1_36_kernel_v_read_readvariableop;
7savev2_adam_dense_layer_1_36_bias_v_read_readvariableop=
9savev2_adam_dense_layer_2_25_kernel_v_read_readvariableop;
7savev2_adam_dense_layer_2_25_bias_v_read_readvariableop<
8savev2_adam_dense_layer_3_3_kernel_v_read_readvariableop:
6savev2_adam_dense_layer_3_3_bias_v_read_readvariableop;
7savev2_adam_logit_probs_36_kernel_v_read_readvariableop9
5savev2_adam_logit_probs_36_bias_v_read_readvariableop
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
value3B1 B+_temp_a671afa322d94957abf6b20af6c4666a/part2	
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*?
value?B?#B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_dense_layer_1_36_kernel_read_readvariableop0savev2_dense_layer_1_36_bias_read_readvariableop2savev2_dense_layer_2_25_kernel_read_readvariableop0savev2_dense_layer_2_25_bias_read_readvariableop1savev2_dense_layer_3_3_kernel_read_readvariableop/savev2_dense_layer_3_3_bias_read_readvariableop0savev2_logit_probs_36_kernel_read_readvariableop.savev2_logit_probs_36_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop9savev2_adam_dense_layer_1_36_kernel_m_read_readvariableop7savev2_adam_dense_layer_1_36_bias_m_read_readvariableop9savev2_adam_dense_layer_2_25_kernel_m_read_readvariableop7savev2_adam_dense_layer_2_25_bias_m_read_readvariableop8savev2_adam_dense_layer_3_3_kernel_m_read_readvariableop6savev2_adam_dense_layer_3_3_bias_m_read_readvariableop7savev2_adam_logit_probs_36_kernel_m_read_readvariableop5savev2_adam_logit_probs_36_bias_m_read_readvariableop9savev2_adam_dense_layer_1_36_kernel_v_read_readvariableop7savev2_adam_dense_layer_1_36_bias_v_read_readvariableop9savev2_adam_dense_layer_2_25_kernel_v_read_readvariableop7savev2_adam_dense_layer_2_25_bias_v_read_readvariableop8savev2_adam_dense_layer_3_3_kernel_v_read_readvariableop6savev2_adam_dense_layer_3_3_bias_v_read_readvariableop7savev2_adam_logit_probs_36_kernel_v_read_readvariableop5savev2_adam_logit_probs_36_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *1
dtypes'
%2#	2
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
??:?:	?d:d:dd:d:d
:
: : : : : : : : : : : :
??:?:	?d:d:dd:d:d
:
:
??:?:	?d:d:dd:d:d
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

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d
: 

_output_shapes
:
:	
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
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?d: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d
: 

_output_shapes
:
:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?d: 

_output_shapes
:d:$  

_output_shapes

:dd: !

_output_shapes
:d:$" 

_output_shapes

:d
: #

_output_shapes
:
:$

_output_shapes
: 
?	
?
+__inference_initModel_layer_call_fn_9445683

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:?????????
**
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_initModel_layer_call_and_return_conditional_losses_94454632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::22
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
: :

_output_shapes
: :

_output_shapes
: 
?
f
G__inference_dropout_47_layer_call_and_return_conditional_losses_9445807

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
J__inference_Dense_layer_1_layer_call_and_return_conditional_losses_9445126

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
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Tanh?
6Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype028
6Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOp?
'Dense_layer_1_36/kernel/Regularizer/AbsAbs>Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2)
'Dense_layer_1_36/kernel/Regularizer/Abs?
)Dense_layer_1_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)Dense_layer_1_36/kernel/Regularizer/Const?
'Dense_layer_1_36/kernel/Regularizer/SumSum+Dense_layer_1_36/kernel/Regularizer/Abs:y:02Dense_layer_1_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/Sum?
)Dense_layer_1_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2+
)Dense_layer_1_36/kernel/Regularizer/mul/x?
'Dense_layer_1_36/kernel/Regularizer/mulMul2Dense_layer_1_36/kernel/Regularizer/mul/x:output:00Dense_layer_1_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/mul?
)Dense_layer_1_36/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_1_36/kernel/Regularizer/add/x?
'Dense_layer_1_36/kernel/Regularizer/addAddV22Dense_layer_1_36/kernel/Regularizer/add/x:output:0+Dense_layer_1_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/add]
IdentityIdentityTanh:y:0*
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
?5
?
F__inference_initModel_layer_call_and_return_conditional_losses_9445641

inputs0
,dense_layer_1_matmul_readvariableop_resource1
-dense_layer_1_biasadd_readvariableop_resource0
,dense_layer_2_matmul_readvariableop_resource1
-dense_layer_2_biasadd_readvariableop_resource0
,dense_layer_3_matmul_readvariableop_resource1
-dense_layer_3_biasadd_readvariableop_resource.
*logit_probs_matmul_readvariableop_resource/
+logit_probs_biasadd_readvariableop_resource
identity?q
dropout_45/IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2
dropout_45/Identity?
#Dense_layer_1/MatMul/ReadVariableOpReadVariableOp,dense_layer_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#Dense_layer_1/MatMul/ReadVariableOp?
Dense_layer_1/MatMulMatMuldropout_45/Identity:output:0+Dense_layer_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_layer_1/MatMul?
$Dense_layer_1/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$Dense_layer_1/BiasAdd/ReadVariableOp?
Dense_layer_1/BiasAddBiasAddDense_layer_1/MatMul:product:0,Dense_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_layer_1/BiasAdd?
Dense_layer_1/TanhTanhDense_layer_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Dense_layer_1/Tanh?
dropout_46/IdentityIdentityDense_layer_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
dropout_46/Identity?
#Dense_layer_2/MatMul/ReadVariableOpReadVariableOp,dense_layer_2_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02%
#Dense_layer_2/MatMul/ReadVariableOp?
Dense_layer_2/MatMulMatMuldropout_46/Identity:output:0+Dense_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
Dense_layer_2/MatMul?
$Dense_layer_2/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02&
$Dense_layer_2/BiasAdd/ReadVariableOp?
Dense_layer_2/BiasAddBiasAddDense_layer_2/MatMul:product:0,Dense_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
Dense_layer_2/BiasAdd?
Dense_layer_2/TanhTanhDense_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Dense_layer_2/Tanh?
dropout_47/IdentityIdentityDense_layer_2/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
dropout_47/Identity?
#Dense_layer_3/MatMul/ReadVariableOpReadVariableOp,dense_layer_3_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02%
#Dense_layer_3/MatMul/ReadVariableOp?
Dense_layer_3/MatMulMatMuldropout_47/Identity:output:0+Dense_layer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
Dense_layer_3/MatMul?
$Dense_layer_3/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02&
$Dense_layer_3/BiasAdd/ReadVariableOp?
Dense_layer_3/BiasAddBiasAddDense_layer_3/MatMul:product:0,Dense_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
Dense_layer_3/BiasAdd?
Dense_layer_3/TanhTanhDense_layer_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Dense_layer_3/Tanh?
dropout_48/IdentityIdentityDense_layer_3/Tanh:y:0*
T0*'
_output_shapes
:?????????d2
dropout_48/Identity?
!Logit_Probs/MatMul/ReadVariableOpReadVariableOp*logit_probs_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02#
!Logit_Probs/MatMul/ReadVariableOp?
Logit_Probs/MatMulMatMuldropout_48/Identity:output:0)Logit_Probs/MatMul/ReadVariableOp:value:0*
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
Logit_Probs/BiasAdd?
6Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp,dense_layer_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype028
6Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOp?
'Dense_layer_1_36/kernel/Regularizer/AbsAbs>Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2)
'Dense_layer_1_36/kernel/Regularizer/Abs?
)Dense_layer_1_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)Dense_layer_1_36/kernel/Regularizer/Const?
'Dense_layer_1_36/kernel/Regularizer/SumSum+Dense_layer_1_36/kernel/Regularizer/Abs:y:02Dense_layer_1_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/Sum?
)Dense_layer_1_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2+
)Dense_layer_1_36/kernel/Regularizer/mul/x?
'Dense_layer_1_36/kernel/Regularizer/mulMul2Dense_layer_1_36/kernel/Regularizer/mul/x:output:00Dense_layer_1_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/mul?
)Dense_layer_1_36/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_1_36/kernel/Regularizer/add/x?
'Dense_layer_1_36/kernel/Regularizer/addAddV22Dense_layer_1_36/kernel/Regularizer/add/x:output:0+Dense_layer_1_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/add?
)Dense_layer_2_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_2_25/kernel/Regularizer/Const?
(Dense_layer_3_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(Dense_layer_3_3/kernel/Regularizer/Constp
IdentityIdentityLogit_Probs/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????:::::::::P L
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
: :

_output_shapes
: :

_output_shapes
: 
?	
?
J__inference_Dense_layer_3_layer_call_and_return_conditional_losses_9445242

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
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
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Tanh?
(Dense_layer_3_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(Dense_layer_3_3/kernel/Regularizer/Const\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:?????????d2

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
?
H
,__inference_dropout_48_layer_call_fn_9445871

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_48_layer_call_and_return_conditional_losses_94452752
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
f
G__inference_dropout_45_layer_call_and_return_conditional_losses_9445695

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
H
,__inference_dropout_46_layer_call_fn_9445773

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
G__inference_dropout_46_layer_call_and_return_conditional_losses_94451592
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
?
-
__inference_loss_fn_1_9445908
identity?
)Dense_layer_2_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_2_25/kernel/Regularizer/Constu
IdentityIdentity2Dense_layer_2_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
f
G__inference_dropout_48_layer_call_and_return_conditional_losses_9445856

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
e
,__inference_dropout_48_layer_call_fn_9445866

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_48_layer_call_and_return_conditional_losses_94452702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
/__inference_Dense_layer_2_layer_call_fn_9445795

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
J__inference_Dense_layer_2_layer_call_and_return_conditional_losses_94451842
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
?
?
J__inference_Dense_layer_1_layer_call_and_return_conditional_losses_9445737

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
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Tanh?
6Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype028
6Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOp?
'Dense_layer_1_36/kernel/Regularizer/AbsAbs>Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2)
'Dense_layer_1_36/kernel/Regularizer/Abs?
)Dense_layer_1_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)Dense_layer_1_36/kernel/Regularizer/Const?
'Dense_layer_1_36/kernel/Regularizer/SumSum+Dense_layer_1_36/kernel/Regularizer/Abs:y:02Dense_layer_1_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/Sum?
)Dense_layer_1_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2+
)Dense_layer_1_36/kernel/Regularizer/mul/x?
'Dense_layer_1_36/kernel/Regularizer/mulMul2Dense_layer_1_36/kernel/Regularizer/mul/x:output:00Dense_layer_1_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/mul?
)Dense_layer_1_36/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_1_36/kernel/Regularizer/add/x?
'Dense_layer_1_36/kernel/Regularizer/addAddV22Dense_layer_1_36/kernel/Regularizer/add/x:output:0+Dense_layer_1_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/add]
IdentityIdentityTanh:y:0*
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
?
e
G__inference_dropout_46_layer_call_and_return_conditional_losses_9445763

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
?2
?
F__inference_initModel_layer_call_and_return_conditional_losses_9445363	
input
dense_layer_1_9445329
dense_layer_1_9445331
dense_layer_2_9445335
dense_layer_2_9445337
dense_layer_3_9445341
dense_layer_3_9445343
logit_probs_9445347
logit_probs_9445349
identity??%Dense_layer_1/StatefulPartitionedCall?%Dense_layer_2/StatefulPartitionedCall?%Dense_layer_3/StatefulPartitionedCall?#Logit_Probs/StatefulPartitionedCall?
dropout_45/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_45_layer_call_and_return_conditional_losses_94450942
dropout_45/PartitionedCall?
%Dense_layer_1/StatefulPartitionedCallStatefulPartitionedCall#dropout_45/PartitionedCall:output:0dense_layer_1_9445329dense_layer_1_9445331*
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
J__inference_Dense_layer_1_layer_call_and_return_conditional_losses_94451262'
%Dense_layer_1/StatefulPartitionedCall?
dropout_46/PartitionedCallPartitionedCall.Dense_layer_1/StatefulPartitionedCall:output:0*
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
G__inference_dropout_46_layer_call_and_return_conditional_losses_94451592
dropout_46/PartitionedCall?
%Dense_layer_2/StatefulPartitionedCallStatefulPartitionedCall#dropout_46/PartitionedCall:output:0dense_layer_2_9445335dense_layer_2_9445337*
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
J__inference_Dense_layer_2_layer_call_and_return_conditional_losses_94451842'
%Dense_layer_2/StatefulPartitionedCall?
dropout_47/PartitionedCallPartitionedCall.Dense_layer_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_47_layer_call_and_return_conditional_losses_94452172
dropout_47/PartitionedCall?
%Dense_layer_3/StatefulPartitionedCallStatefulPartitionedCall#dropout_47/PartitionedCall:output:0dense_layer_3_9445341dense_layer_3_9445343*
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
J__inference_Dense_layer_3_layer_call_and_return_conditional_losses_94452422'
%Dense_layer_3/StatefulPartitionedCall?
dropout_48/PartitionedCallPartitionedCall.Dense_layer_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_48_layer_call_and_return_conditional_losses_94452752
dropout_48/PartitionedCall?
#Logit_Probs/StatefulPartitionedCallStatefulPartitionedCall#dropout_48/PartitionedCall:output:0logit_probs_9445347logit_probs_9445349*
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
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_94452982%
#Logit_Probs/StatefulPartitionedCall?
6Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_layer_1_9445329* 
_output_shapes
:
??*
dtype028
6Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOp?
'Dense_layer_1_36/kernel/Regularizer/AbsAbs>Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2)
'Dense_layer_1_36/kernel/Regularizer/Abs?
)Dense_layer_1_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)Dense_layer_1_36/kernel/Regularizer/Const?
'Dense_layer_1_36/kernel/Regularizer/SumSum+Dense_layer_1_36/kernel/Regularizer/Abs:y:02Dense_layer_1_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/Sum?
)Dense_layer_1_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2+
)Dense_layer_1_36/kernel/Regularizer/mul/x?
'Dense_layer_1_36/kernel/Regularizer/mulMul2Dense_layer_1_36/kernel/Regularizer/mul/x:output:00Dense_layer_1_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/mul?
)Dense_layer_1_36/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_1_36/kernel/Regularizer/add/x?
'Dense_layer_1_36/kernel/Regularizer/addAddV22Dense_layer_1_36/kernel/Regularizer/add/x:output:0+Dense_layer_1_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/add?
)Dense_layer_2_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_2_25/kernel/Regularizer/Const?
(Dense_layer_3_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(Dense_layer_3_3/kernel/Regularizer/Const?
IdentityIdentity,Logit_Probs/StatefulPartitionedCall:output:0&^Dense_layer_1/StatefulPartitionedCall&^Dense_layer_2/StatefulPartitionedCall&^Dense_layer_3/StatefulPartitionedCall$^Logit_Probs/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::2N
%Dense_layer_1/StatefulPartitionedCall%Dense_layer_1/StatefulPartitionedCall2N
%Dense_layer_2/StatefulPartitionedCall%Dense_layer_2/StatefulPartitionedCall2N
%Dense_layer_3/StatefulPartitionedCall%Dense_layer_3/StatefulPartitionedCall2J
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
: :

_output_shapes
: :

_output_shapes
: 
?	
?
+__inference_initModel_layer_call_fn_9445423	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:?????????
**
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_initModel_layer_call_and_return_conditional_losses_94454042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::22
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
: :

_output_shapes
: :

_output_shapes
: 
?
e
,__inference_dropout_47_layer_call_fn_9445817

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_47_layer_call_and_return_conditional_losses_94452122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
%__inference_signature_wrapper_9445523	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:?????????
**
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*+
f&R$
"__inference__wrapped_model_94450732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::22
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
: :

_output_shapes
: :

_output_shapes
: 
?
?
/__inference_Dense_layer_1_layer_call_fn_9445746

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
J__inference_Dense_layer_1_layer_call_and_return_conditional_losses_94451262
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
?
e
G__inference_dropout_45_layer_call_and_return_conditional_losses_9445094

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_47_layer_call_and_return_conditional_losses_9445812

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????d2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????d2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
+__inference_initModel_layer_call_fn_9445482	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:?????????
**
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_initModel_layer_call_and_return_conditional_losses_94454632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::22
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
: :

_output_shapes
: :

_output_shapes
: 
Ӛ
?
#__inference__traced_restore_9446162
file_prefix,
(assignvariableop_dense_layer_1_36_kernel,
(assignvariableop_1_dense_layer_1_36_bias.
*assignvariableop_2_dense_layer_2_25_kernel,
(assignvariableop_3_dense_layer_2_25_bias-
)assignvariableop_4_dense_layer_3_3_kernel+
'assignvariableop_5_dense_layer_3_3_bias,
(assignvariableop_6_logit_probs_36_kernel*
&assignvariableop_7_logit_probs_36_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_1
assignvariableop_17_total_2
assignvariableop_18_count_26
2assignvariableop_19_adam_dense_layer_1_36_kernel_m4
0assignvariableop_20_adam_dense_layer_1_36_bias_m6
2assignvariableop_21_adam_dense_layer_2_25_kernel_m4
0assignvariableop_22_adam_dense_layer_2_25_bias_m5
1assignvariableop_23_adam_dense_layer_3_3_kernel_m3
/assignvariableop_24_adam_dense_layer_3_3_bias_m4
0assignvariableop_25_adam_logit_probs_36_kernel_m2
.assignvariableop_26_adam_logit_probs_36_bias_m6
2assignvariableop_27_adam_dense_layer_1_36_kernel_v4
0assignvariableop_28_adam_dense_layer_1_36_bias_v6
2assignvariableop_29_adam_dense_layer_2_25_kernel_v4
0assignvariableop_30_adam_dense_layer_2_25_bias_v5
1assignvariableop_31_adam_dense_layer_3_3_kernel_v3
/assignvariableop_32_adam_dense_layer_3_3_bias_v4
0assignvariableop_33_adam_logit_probs_36_kernel_v2
.assignvariableop_34_adam_logit_probs_36_bias_v
identity_36??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*?
value?B?#B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::*1
dtypes'
%2#	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp(assignvariableop_dense_layer_1_36_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp(assignvariableop_1_dense_layer_1_36_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp*assignvariableop_2_dense_layer_2_25_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp(assignvariableop_3_dense_layer_2_25_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp)assignvariableop_4_dense_layer_3_3_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp'assignvariableop_5_dense_layer_3_3_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp(assignvariableop_6_logit_probs_36_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp&assignvariableop_7_logit_probs_36_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0	*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_2Identity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_2Identity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp2assignvariableop_19_adam_dense_layer_1_36_kernel_mIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp0assignvariableop_20_adam_dense_layer_1_36_bias_mIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp2assignvariableop_21_adam_dense_layer_2_25_kernel_mIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp0assignvariableop_22_adam_dense_layer_2_25_bias_mIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp1assignvariableop_23_adam_dense_layer_3_3_kernel_mIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp/assignvariableop_24_adam_dense_layer_3_3_bias_mIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp0assignvariableop_25_adam_logit_probs_36_kernel_mIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp.assignvariableop_26_adam_logit_probs_36_bias_mIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp2assignvariableop_27_adam_dense_layer_1_36_kernel_vIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp0assignvariableop_28_adam_dense_layer_1_36_bias_vIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp2assignvariableop_29_adam_dense_layer_2_25_kernel_vIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp0assignvariableop_30_adam_dense_layer_2_25_bias_vIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp1assignvariableop_31_adam_dense_layer_3_3_kernel_vIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp/assignvariableop_32_adam_dense_layer_3_3_bias_vIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp0assignvariableop_33_adam_logit_probs_36_kernel_vIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp.assignvariableop_34_adam_logit_probs_36_bias_vIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34?
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
NoOp?
Identity_35Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_35?
Identity_36IdentityIdentity_35:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_36"#
identity_36Identity_36:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_34AssignVariableOp_342(
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
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: 
?
-
__inference_loss_fn_2_9445913
identity?
(Dense_layer_3_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(Dense_layer_3_3/kernel/Regularizer/Constt
IdentityIdentity1Dense_layer_3_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
?
-__inference_Logit_Probs_layer_call_fn_9445890

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
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_94452982
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
: 
?
e
G__inference_dropout_48_layer_call_and_return_conditional_losses_9445861

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????d2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????d2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_9445298

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
?
e
G__inference_dropout_45_layer_call_and_return_conditional_losses_9445700

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_Dense_layer_3_layer_call_fn_9445844

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
J__inference_Dense_layer_3_layer_call_and_return_conditional_losses_94452422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

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
: 
?
f
G__inference_dropout_47_layer_call_and_return_conditional_losses_9445212

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?2
?
F__inference_initModel_layer_call_and_return_conditional_losses_9445463

inputs
dense_layer_1_9445429
dense_layer_1_9445431
dense_layer_2_9445435
dense_layer_2_9445437
dense_layer_3_9445441
dense_layer_3_9445443
logit_probs_9445447
logit_probs_9445449
identity??%Dense_layer_1/StatefulPartitionedCall?%Dense_layer_2/StatefulPartitionedCall?%Dense_layer_3/StatefulPartitionedCall?#Logit_Probs/StatefulPartitionedCall?
dropout_45/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_45_layer_call_and_return_conditional_losses_94450942
dropout_45/PartitionedCall?
%Dense_layer_1/StatefulPartitionedCallStatefulPartitionedCall#dropout_45/PartitionedCall:output:0dense_layer_1_9445429dense_layer_1_9445431*
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
J__inference_Dense_layer_1_layer_call_and_return_conditional_losses_94451262'
%Dense_layer_1/StatefulPartitionedCall?
dropout_46/PartitionedCallPartitionedCall.Dense_layer_1/StatefulPartitionedCall:output:0*
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
G__inference_dropout_46_layer_call_and_return_conditional_losses_94451592
dropout_46/PartitionedCall?
%Dense_layer_2/StatefulPartitionedCallStatefulPartitionedCall#dropout_46/PartitionedCall:output:0dense_layer_2_9445435dense_layer_2_9445437*
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
J__inference_Dense_layer_2_layer_call_and_return_conditional_losses_94451842'
%Dense_layer_2/StatefulPartitionedCall?
dropout_47/PartitionedCallPartitionedCall.Dense_layer_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_47_layer_call_and_return_conditional_losses_94452172
dropout_47/PartitionedCall?
%Dense_layer_3/StatefulPartitionedCallStatefulPartitionedCall#dropout_47/PartitionedCall:output:0dense_layer_3_9445441dense_layer_3_9445443*
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
J__inference_Dense_layer_3_layer_call_and_return_conditional_losses_94452422'
%Dense_layer_3/StatefulPartitionedCall?
dropout_48/PartitionedCallPartitionedCall.Dense_layer_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_48_layer_call_and_return_conditional_losses_94452752
dropout_48/PartitionedCall?
#Logit_Probs/StatefulPartitionedCallStatefulPartitionedCall#dropout_48/PartitionedCall:output:0logit_probs_9445447logit_probs_9445449*
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
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_94452982%
#Logit_Probs/StatefulPartitionedCall?
6Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_layer_1_9445429* 
_output_shapes
:
??*
dtype028
6Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOp?
'Dense_layer_1_36/kernel/Regularizer/AbsAbs>Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2)
'Dense_layer_1_36/kernel/Regularizer/Abs?
)Dense_layer_1_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)Dense_layer_1_36/kernel/Regularizer/Const?
'Dense_layer_1_36/kernel/Regularizer/SumSum+Dense_layer_1_36/kernel/Regularizer/Abs:y:02Dense_layer_1_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/Sum?
)Dense_layer_1_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2+
)Dense_layer_1_36/kernel/Regularizer/mul/x?
'Dense_layer_1_36/kernel/Regularizer/mulMul2Dense_layer_1_36/kernel/Regularizer/mul/x:output:00Dense_layer_1_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/mul?
)Dense_layer_1_36/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_1_36/kernel/Regularizer/add/x?
'Dense_layer_1_36/kernel/Regularizer/addAddV22Dense_layer_1_36/kernel/Regularizer/add/x:output:0+Dense_layer_1_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/add?
)Dense_layer_2_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_2_25/kernel/Regularizer/Const?
(Dense_layer_3_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(Dense_layer_3_3/kernel/Regularizer/Const?
IdentityIdentity,Logit_Probs/StatefulPartitionedCall:output:0&^Dense_layer_1/StatefulPartitionedCall&^Dense_layer_2/StatefulPartitionedCall&^Dense_layer_3/StatefulPartitionedCall$^Logit_Probs/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::2N
%Dense_layer_1/StatefulPartitionedCall%Dense_layer_1/StatefulPartitionedCall2N
%Dense_layer_2/StatefulPartitionedCall%Dense_layer_2/StatefulPartitionedCall2N
%Dense_layer_3/StatefulPartitionedCall%Dense_layer_3/StatefulPartitionedCall2J
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
: :

_output_shapes
: :

_output_shapes
: 
?
H
,__inference_dropout_45_layer_call_fn_9445710

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_45_layer_call_and_return_conditional_losses_94450942
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
J__inference_Dense_layer_2_layer_call_and_return_conditional_losses_9445184

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
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Tanh?
)Dense_layer_2_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_2_25/kernel/Regularizer/Const\
IdentityIdentityTanh:y:0*
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
?

?
J__inference_Dense_layer_2_layer_call_and_return_conditional_losses_9445786

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
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Tanh?
)Dense_layer_2_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_2_25/kernel/Regularizer/Const\
IdentityIdentityTanh:y:0*
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
?9
?
F__inference_initModel_layer_call_and_return_conditional_losses_9445404

inputs
dense_layer_1_9445370
dense_layer_1_9445372
dense_layer_2_9445376
dense_layer_2_9445378
dense_layer_3_9445382
dense_layer_3_9445384
logit_probs_9445388
logit_probs_9445390
identity??%Dense_layer_1/StatefulPartitionedCall?%Dense_layer_2/StatefulPartitionedCall?%Dense_layer_3/StatefulPartitionedCall?#Logit_Probs/StatefulPartitionedCall?"dropout_45/StatefulPartitionedCall?"dropout_46/StatefulPartitionedCall?"dropout_47/StatefulPartitionedCall?"dropout_48/StatefulPartitionedCall?
"dropout_45/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_45_layer_call_and_return_conditional_losses_94450892$
"dropout_45/StatefulPartitionedCall?
%Dense_layer_1/StatefulPartitionedCallStatefulPartitionedCall+dropout_45/StatefulPartitionedCall:output:0dense_layer_1_9445370dense_layer_1_9445372*
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
J__inference_Dense_layer_1_layer_call_and_return_conditional_losses_94451262'
%Dense_layer_1/StatefulPartitionedCall?
"dropout_46/StatefulPartitionedCallStatefulPartitionedCall.Dense_layer_1/StatefulPartitionedCall:output:0#^dropout_45/StatefulPartitionedCall*
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
G__inference_dropout_46_layer_call_and_return_conditional_losses_94451542$
"dropout_46/StatefulPartitionedCall?
%Dense_layer_2/StatefulPartitionedCallStatefulPartitionedCall+dropout_46/StatefulPartitionedCall:output:0dense_layer_2_9445376dense_layer_2_9445378*
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
J__inference_Dense_layer_2_layer_call_and_return_conditional_losses_94451842'
%Dense_layer_2/StatefulPartitionedCall?
"dropout_47/StatefulPartitionedCallStatefulPartitionedCall.Dense_layer_2/StatefulPartitionedCall:output:0#^dropout_46/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_47_layer_call_and_return_conditional_losses_94452122$
"dropout_47/StatefulPartitionedCall?
%Dense_layer_3/StatefulPartitionedCallStatefulPartitionedCall+dropout_47/StatefulPartitionedCall:output:0dense_layer_3_9445382dense_layer_3_9445384*
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
J__inference_Dense_layer_3_layer_call_and_return_conditional_losses_94452422'
%Dense_layer_3/StatefulPartitionedCall?
"dropout_48/StatefulPartitionedCallStatefulPartitionedCall.Dense_layer_3/StatefulPartitionedCall:output:0#^dropout_47/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_48_layer_call_and_return_conditional_losses_94452702$
"dropout_48/StatefulPartitionedCall?
#Logit_Probs/StatefulPartitionedCallStatefulPartitionedCall+dropout_48/StatefulPartitionedCall:output:0logit_probs_9445388logit_probs_9445390*
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
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_94452982%
#Logit_Probs/StatefulPartitionedCall?
6Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_layer_1_9445370* 
_output_shapes
:
??*
dtype028
6Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOp?
'Dense_layer_1_36/kernel/Regularizer/AbsAbs>Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2)
'Dense_layer_1_36/kernel/Regularizer/Abs?
)Dense_layer_1_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)Dense_layer_1_36/kernel/Regularizer/Const?
'Dense_layer_1_36/kernel/Regularizer/SumSum+Dense_layer_1_36/kernel/Regularizer/Abs:y:02Dense_layer_1_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/Sum?
)Dense_layer_1_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2+
)Dense_layer_1_36/kernel/Regularizer/mul/x?
'Dense_layer_1_36/kernel/Regularizer/mulMul2Dense_layer_1_36/kernel/Regularizer/mul/x:output:00Dense_layer_1_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/mul?
)Dense_layer_1_36/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_1_36/kernel/Regularizer/add/x?
'Dense_layer_1_36/kernel/Regularizer/addAddV22Dense_layer_1_36/kernel/Regularizer/add/x:output:0+Dense_layer_1_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/add?
)Dense_layer_2_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_2_25/kernel/Regularizer/Const?
(Dense_layer_3_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(Dense_layer_3_3/kernel/Regularizer/Const?
IdentityIdentity,Logit_Probs/StatefulPartitionedCall:output:0&^Dense_layer_1/StatefulPartitionedCall&^Dense_layer_2/StatefulPartitionedCall&^Dense_layer_3/StatefulPartitionedCall$^Logit_Probs/StatefulPartitionedCall#^dropout_45/StatefulPartitionedCall#^dropout_46/StatefulPartitionedCall#^dropout_47/StatefulPartitionedCall#^dropout_48/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::2N
%Dense_layer_1/StatefulPartitionedCall%Dense_layer_1/StatefulPartitionedCall2N
%Dense_layer_2/StatefulPartitionedCall%Dense_layer_2/StatefulPartitionedCall2N
%Dense_layer_3/StatefulPartitionedCall%Dense_layer_3/StatefulPartitionedCall2J
#Logit_Probs/StatefulPartitionedCall#Logit_Probs/StatefulPartitionedCall2H
"dropout_45/StatefulPartitionedCall"dropout_45/StatefulPartitionedCall2H
"dropout_46/StatefulPartitionedCall"dropout_46/StatefulPartitionedCall2H
"dropout_47/StatefulPartitionedCall"dropout_47/StatefulPartitionedCall2H
"dropout_48/StatefulPartitionedCall"dropout_48/StatefulPartitionedCall:P L
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
: :

_output_shapes
: :

_output_shapes
: 
?
e
,__inference_dropout_45_layer_call_fn_9445705

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_45_layer_call_and_return_conditional_losses_94450892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
G__inference_dropout_46_layer_call_and_return_conditional_losses_9445154

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
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
 *    2
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
?
u
__inference_loss_fn_0_9445903C
?dense_layer_1_36_kernel_regularizer_abs_readvariableop_resource
identity??
6Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp?dense_layer_1_36_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
??*
dtype028
6Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOp?
'Dense_layer_1_36/kernel/Regularizer/AbsAbs>Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2)
'Dense_layer_1_36/kernel/Regularizer/Abs?
)Dense_layer_1_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)Dense_layer_1_36/kernel/Regularizer/Const?
'Dense_layer_1_36/kernel/Regularizer/SumSum+Dense_layer_1_36/kernel/Regularizer/Abs:y:02Dense_layer_1_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/Sum?
)Dense_layer_1_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2+
)Dense_layer_1_36/kernel/Regularizer/mul/x?
'Dense_layer_1_36/kernel/Regularizer/mulMul2Dense_layer_1_36/kernel/Regularizer/mul/x:output:00Dense_layer_1_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/mul?
)Dense_layer_1_36/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_1_36/kernel/Regularizer/add/x?
'Dense_layer_1_36/kernel/Regularizer/addAddV22Dense_layer_1_36/kernel/Regularizer/add/x:output:0+Dense_layer_1_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/addn
IdentityIdentity+Dense_layer_1_36/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
?
f
G__inference_dropout_46_layer_call_and_return_conditional_losses_9445758

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
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
 *    2
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
?
e
G__inference_dropout_47_layer_call_and_return_conditional_losses_9445217

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????d2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????d2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
H
,__inference_dropout_47_layer_call_fn_9445822

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_47_layer_call_and_return_conditional_losses_94452172
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
e
G__inference_dropout_46_layer_call_and_return_conditional_losses_9445159

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
?
e
,__inference_dropout_46_layer_call_fn_9445768

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
G__inference_dropout_46_layer_call_and_return_conditional_losses_94451542
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
?9
?
F__inference_initModel_layer_call_and_return_conditional_losses_9445325	
input
dense_layer_1_9445137
dense_layer_1_9445139
dense_layer_2_9445195
dense_layer_2_9445197
dense_layer_3_9445253
dense_layer_3_9445255
logit_probs_9445309
logit_probs_9445311
identity??%Dense_layer_1/StatefulPartitionedCall?%Dense_layer_2/StatefulPartitionedCall?%Dense_layer_3/StatefulPartitionedCall?#Logit_Probs/StatefulPartitionedCall?"dropout_45/StatefulPartitionedCall?"dropout_46/StatefulPartitionedCall?"dropout_47/StatefulPartitionedCall?"dropout_48/StatefulPartitionedCall?
"dropout_45/StatefulPartitionedCallStatefulPartitionedCallinput*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_45_layer_call_and_return_conditional_losses_94450892$
"dropout_45/StatefulPartitionedCall?
%Dense_layer_1/StatefulPartitionedCallStatefulPartitionedCall+dropout_45/StatefulPartitionedCall:output:0dense_layer_1_9445137dense_layer_1_9445139*
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
J__inference_Dense_layer_1_layer_call_and_return_conditional_losses_94451262'
%Dense_layer_1/StatefulPartitionedCall?
"dropout_46/StatefulPartitionedCallStatefulPartitionedCall.Dense_layer_1/StatefulPartitionedCall:output:0#^dropout_45/StatefulPartitionedCall*
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
G__inference_dropout_46_layer_call_and_return_conditional_losses_94451542$
"dropout_46/StatefulPartitionedCall?
%Dense_layer_2/StatefulPartitionedCallStatefulPartitionedCall+dropout_46/StatefulPartitionedCall:output:0dense_layer_2_9445195dense_layer_2_9445197*
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
J__inference_Dense_layer_2_layer_call_and_return_conditional_losses_94451842'
%Dense_layer_2/StatefulPartitionedCall?
"dropout_47/StatefulPartitionedCallStatefulPartitionedCall.Dense_layer_2/StatefulPartitionedCall:output:0#^dropout_46/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_47_layer_call_and_return_conditional_losses_94452122$
"dropout_47/StatefulPartitionedCall?
%Dense_layer_3/StatefulPartitionedCallStatefulPartitionedCall+dropout_47/StatefulPartitionedCall:output:0dense_layer_3_9445253dense_layer_3_9445255*
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
J__inference_Dense_layer_3_layer_call_and_return_conditional_losses_94452422'
%Dense_layer_3/StatefulPartitionedCall?
"dropout_48/StatefulPartitionedCallStatefulPartitionedCall.Dense_layer_3/StatefulPartitionedCall:output:0#^dropout_47/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_48_layer_call_and_return_conditional_losses_94452702$
"dropout_48/StatefulPartitionedCall?
#Logit_Probs/StatefulPartitionedCallStatefulPartitionedCall+dropout_48/StatefulPartitionedCall:output:0logit_probs_9445309logit_probs_9445311*
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
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_94452982%
#Logit_Probs/StatefulPartitionedCall?
6Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_layer_1_9445137* 
_output_shapes
:
??*
dtype028
6Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOp?
'Dense_layer_1_36/kernel/Regularizer/AbsAbs>Dense_layer_1_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2)
'Dense_layer_1_36/kernel/Regularizer/Abs?
)Dense_layer_1_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)Dense_layer_1_36/kernel/Regularizer/Const?
'Dense_layer_1_36/kernel/Regularizer/SumSum+Dense_layer_1_36/kernel/Regularizer/Abs:y:02Dense_layer_1_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/Sum?
)Dense_layer_1_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2+
)Dense_layer_1_36/kernel/Regularizer/mul/x?
'Dense_layer_1_36/kernel/Regularizer/mulMul2Dense_layer_1_36/kernel/Regularizer/mul/x:output:00Dense_layer_1_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/mul?
)Dense_layer_1_36/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_1_36/kernel/Regularizer/add/x?
'Dense_layer_1_36/kernel/Regularizer/addAddV22Dense_layer_1_36/kernel/Regularizer/add/x:output:0+Dense_layer_1_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_36/kernel/Regularizer/add?
)Dense_layer_2_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_2_25/kernel/Regularizer/Const?
(Dense_layer_3_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(Dense_layer_3_3/kernel/Regularizer/Const?
IdentityIdentity,Logit_Probs/StatefulPartitionedCall:output:0&^Dense_layer_1/StatefulPartitionedCall&^Dense_layer_2/StatefulPartitionedCall&^Dense_layer_3/StatefulPartitionedCall$^Logit_Probs/StatefulPartitionedCall#^dropout_45/StatefulPartitionedCall#^dropout_46/StatefulPartitionedCall#^dropout_47/StatefulPartitionedCall#^dropout_48/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::2N
%Dense_layer_1/StatefulPartitionedCall%Dense_layer_1/StatefulPartitionedCall2N
%Dense_layer_2/StatefulPartitionedCall%Dense_layer_2/StatefulPartitionedCall2N
%Dense_layer_3/StatefulPartitionedCall%Dense_layer_3/StatefulPartitionedCall2J
#Logit_Probs/StatefulPartitionedCall#Logit_Probs/StatefulPartitionedCall2H
"dropout_45/StatefulPartitionedCall"dropout_45/StatefulPartitionedCall2H
"dropout_46/StatefulPartitionedCall"dropout_46/StatefulPartitionedCall2H
"dropout_47/StatefulPartitionedCall"dropout_47/StatefulPartitionedCall2H
"dropout_48/StatefulPartitionedCall"dropout_48/StatefulPartitionedCall:O K
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
: :

_output_shapes
: :

_output_shapes
: 
?
e
G__inference_dropout_48_layer_call_and_return_conditional_losses_9445275

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????d2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????d2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
+__inference_initModel_layer_call_fn_9445662

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:?????????
**
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_initModel_layer_call_and_return_conditional_losses_94454042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::22
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
: :

_output_shapes
: :

_output_shapes
: 
?	
?
J__inference_Dense_layer_3_layer_call_and_return_conditional_losses_9445835

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
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
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Tanh?
(Dense_layer_3_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(Dense_layer_3_3/kernel/Regularizer/Const\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:?????????d2

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
?
f
G__inference_dropout_48_layer_call_and_return_conditional_losses_9445270

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs"?L
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
tensorflow/serving/predict:??
?@
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?<
_tf_keras_model?<{"class_name": "Model", "name": "initModel", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "initModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}, "name": "Input", "inbound_nodes": []}, {"class_name": "Dropout", "config": {"name": "dropout_45", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_45", "inbound_nodes": [[["Input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_layer_1", "trainable": true, "dtype": "float32", "units": 200, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0010000000474974513, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_layer_1", "inbound_nodes": [[["dropout_45", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_46", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_46", "inbound_nodes": [[["Dense_layer_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_layer_2", "trainable": true, "dtype": "float32", "units": 100, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_layer_2", "inbound_nodes": [[["dropout_46", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_47", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_47", "inbound_nodes": [[["Dense_layer_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_layer_3", "trainable": true, "dtype": "float32", "units": 100, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_layer_3", "inbound_nodes": [[["dropout_47", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_48", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_48", "inbound_nodes": [[["Dense_layer_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Logit_Probs", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Logit_Probs", "inbound_nodes": [[["dropout_48", 0, 0, {}]]]}], "input_layers": [["Input", 0, 0]], "output_layers": [["Logit_Probs", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "initModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}, "name": "Input", "inbound_nodes": []}, {"class_name": "Dropout", "config": {"name": "dropout_45", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_45", "inbound_nodes": [[["Input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_layer_1", "trainable": true, "dtype": "float32", "units": 200, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0010000000474974513, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_layer_1", "inbound_nodes": [[["dropout_45", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_46", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_46", "inbound_nodes": [[["Dense_layer_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_layer_2", "trainable": true, "dtype": "float32", "units": 100, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_layer_2", "inbound_nodes": [[["dropout_46", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_47", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_47", "inbound_nodes": [[["Dense_layer_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_layer_3", "trainable": true, "dtype": "float32", "units": 100, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_layer_3", "inbound_nodes": [[["dropout_47", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_48", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_48", "inbound_nodes": [[["Dense_layer_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Logit_Probs", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Logit_Probs", "inbound_nodes": [[["dropout_48", 0, 0, {}]]]}], "input_layers": [["Input", 0, 0]], "output_layers": [["Logit_Probs", 0, 0]]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}}, "metrics": ["accuracy", {"class_name": "SparseTopKCategoricalAccuracy", "config": {"name": "sparse_top_k_categorical_accuracy", "dtype": "float32", "k": 3}}], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "Input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}}
?
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_45", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_45", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "Dense_layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Dense_layer_1", "trainable": true, "dtype": "float32", "units": 200, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0010000000474974513, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
?
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_46", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_46", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
?

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "Dense_layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Dense_layer_2", "trainable": true, "dtype": "float32", "units": 100, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
?
$regularization_losses
%trainable_variables
&	variables
'	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_47", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_47", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
?

(kernel
)bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "Dense_layer_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Dense_layer_3", "trainable": true, "dtype": "float32", "units": 100, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?
.regularization_losses
/trainable_variables
0	variables
1	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_48", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_48", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
?

2kernel
3bias
4regularization_losses
5trainable_variables
6	variables
7	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "Logit_Probs", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Logit_Probs", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?
8iter

9beta_1

:beta_2
	;decay
<learning_ratem{m|m}m~(m)m?2m?3m?v?v?v?v?(v?)v?2v?3v?"
	optimizer
8
?0
?1
?2"
trackable_list_wrapper
X
0
1
2
3
(4
)5
26
37"
trackable_list_wrapper
X
0
1
2
3
(4
)5
26
37"
trackable_list_wrapper
?
regularization_losses

=layers
trainable_variables
	variables
>layer_regularization_losses
?layer_metrics
@metrics
Anon_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses

Blayers
Clayer_regularization_losses
trainable_variables
	variables
Dlayer_metrics
Emetrics
Fnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)
??2Dense_layer_1_36/kernel
$:"?2Dense_layer_1_36/bias
(
?0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses

Glayers
Hlayer_regularization_losses
trainable_variables
	variables
Ilayer_metrics
Jmetrics
Knon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses

Llayers
Mlayer_regularization_losses
trainable_variables
	variables
Nlayer_metrics
Ometrics
Pnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(	?d2Dense_layer_2_25/kernel
#:!d2Dense_layer_2_25/bias
(
?0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
 regularization_losses

Qlayers
Rlayer_regularization_losses
!trainable_variables
"	variables
Slayer_metrics
Tmetrics
Unon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
$regularization_losses

Vlayers
Wlayer_regularization_losses
%trainable_variables
&	variables
Xlayer_metrics
Ymetrics
Znon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&dd2Dense_layer_3_3/kernel
": d2Dense_layer_3_3/bias
(
?0"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?
*regularization_losses

[layers
\layer_regularization_losses
+trainable_variables
,	variables
]layer_metrics
^metrics
_non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
.regularization_losses

`layers
alayer_regularization_losses
/trainable_variables
0	variables
blayer_metrics
cmetrics
dnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%d
2Logit_Probs_36/kernel
!:
2Logit_Probs_36/bias
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
?
4regularization_losses

elayers
flayer_regularization_losses
5trainable_variables
6	variables
glayer_metrics
hmetrics
inon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
j0
k1
l2"
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
(
?0"
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
(
?0"
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
(
?0"
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
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	qtotal
	rcount
s
_fn_kwargs
t	variables
u	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
?
	vtotal
	wcount
x
_fn_kwargs
y	variables
z	keras_api"?
_tf_keras_metric?{"class_name": "SparseTopKCategoricalAccuracy", "name": "sparse_top_k_categorical_accuracy", "dtype": "float32", "config": {"name": "sparse_top_k_categorical_accuracy", "dtype": "float32", "k": 3}}
:  (2total
:  (2count
.
m0
n1"
trackable_list_wrapper
-
o	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
q0
r1"
trackable_list_wrapper
-
t	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
v0
w1"
trackable_list_wrapper
-
y	variables"
_generic_user_object
0:.
??2Adam/Dense_layer_1_36/kernel/m
):'?2Adam/Dense_layer_1_36/bias/m
/:-	?d2Adam/Dense_layer_2_25/kernel/m
(:&d2Adam/Dense_layer_2_25/bias/m
-:+dd2Adam/Dense_layer_3_3/kernel/m
':%d2Adam/Dense_layer_3_3/bias/m
,:*d
2Adam/Logit_Probs_36/kernel/m
&:$
2Adam/Logit_Probs_36/bias/m
0:.
??2Adam/Dense_layer_1_36/kernel/v
):'?2Adam/Dense_layer_1_36/bias/v
/:-	?d2Adam/Dense_layer_2_25/kernel/v
(:&d2Adam/Dense_layer_2_25/bias/v
-:+dd2Adam/Dense_layer_3_3/kernel/v
':%d2Adam/Dense_layer_3_3/bias/v
,:*d
2Adam/Logit_Probs_36/kernel/v
&:$
2Adam/Logit_Probs_36/bias/v
?2?
"__inference__wrapped_model_9445073?
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
F__inference_initModel_layer_call_and_return_conditional_losses_9445641
F__inference_initModel_layer_call_and_return_conditional_losses_9445596
F__inference_initModel_layer_call_and_return_conditional_losses_9445363
F__inference_initModel_layer_call_and_return_conditional_losses_9445325?
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
+__inference_initModel_layer_call_fn_9445683
+__inference_initModel_layer_call_fn_9445662
+__inference_initModel_layer_call_fn_9445423
+__inference_initModel_layer_call_fn_9445482?
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
G__inference_dropout_45_layer_call_and_return_conditional_losses_9445700
G__inference_dropout_45_layer_call_and_return_conditional_losses_9445695?
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
,__inference_dropout_45_layer_call_fn_9445710
,__inference_dropout_45_layer_call_fn_9445705?
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
J__inference_Dense_layer_1_layer_call_and_return_conditional_losses_9445737?
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
/__inference_Dense_layer_1_layer_call_fn_9445746?
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
G__inference_dropout_46_layer_call_and_return_conditional_losses_9445758
G__inference_dropout_46_layer_call_and_return_conditional_losses_9445763?
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
,__inference_dropout_46_layer_call_fn_9445768
,__inference_dropout_46_layer_call_fn_9445773?
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
J__inference_Dense_layer_2_layer_call_and_return_conditional_losses_9445786?
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
/__inference_Dense_layer_2_layer_call_fn_9445795?
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
G__inference_dropout_47_layer_call_and_return_conditional_losses_9445812
G__inference_dropout_47_layer_call_and_return_conditional_losses_9445807?
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
,__inference_dropout_47_layer_call_fn_9445822
,__inference_dropout_47_layer_call_fn_9445817?
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
J__inference_Dense_layer_3_layer_call_and_return_conditional_losses_9445835?
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
/__inference_Dense_layer_3_layer_call_fn_9445844?
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
G__inference_dropout_48_layer_call_and_return_conditional_losses_9445861
G__inference_dropout_48_layer_call_and_return_conditional_losses_9445856?
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
,__inference_dropout_48_layer_call_fn_9445866
,__inference_dropout_48_layer_call_fn_9445871?
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
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_9445881?
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
-__inference_Logit_Probs_layer_call_fn_9445890?
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
__inference_loss_fn_0_9445903?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_9445908?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_9445913?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
2B0
%__inference_signature_wrapper_9445523Input?
J__inference_Dense_layer_1_layer_call_and_return_conditional_losses_9445737^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
/__inference_Dense_layer_1_layer_call_fn_9445746Q0?-
&?#
!?
inputs??????????
? "????????????
J__inference_Dense_layer_2_layer_call_and_return_conditional_losses_9445786]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????d
? ?
/__inference_Dense_layer_2_layer_call_fn_9445795P0?-
&?#
!?
inputs??????????
? "??????????d?
J__inference_Dense_layer_3_layer_call_and_return_conditional_losses_9445835\()/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????d
? ?
/__inference_Dense_layer_3_layer_call_fn_9445844O()/?,
%?"
 ?
inputs?????????d
? "??????????d?
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_9445881\23/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????

? ?
-__inference_Logit_Probs_layer_call_fn_9445890O23/?,
%?"
 ?
inputs?????????d
? "??????????
?
"__inference__wrapped_model_9445073v()23/?,
%?"
 ?
Input??????????
? "9?6
4
Logit_Probs%?"
Logit_Probs?????????
?
G__inference_dropout_45_layer_call_and_return_conditional_losses_9445695^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
G__inference_dropout_45_layer_call_and_return_conditional_losses_9445700^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
,__inference_dropout_45_layer_call_fn_9445705Q4?1
*?'
!?
inputs??????????
p
? "????????????
,__inference_dropout_45_layer_call_fn_9445710Q4?1
*?'
!?
inputs??????????
p 
? "????????????
G__inference_dropout_46_layer_call_and_return_conditional_losses_9445758^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
G__inference_dropout_46_layer_call_and_return_conditional_losses_9445763^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
,__inference_dropout_46_layer_call_fn_9445768Q4?1
*?'
!?
inputs??????????
p
? "????????????
,__inference_dropout_46_layer_call_fn_9445773Q4?1
*?'
!?
inputs??????????
p 
? "????????????
G__inference_dropout_47_layer_call_and_return_conditional_losses_9445807\3?0
)?&
 ?
inputs?????????d
p
? "%?"
?
0?????????d
? ?
G__inference_dropout_47_layer_call_and_return_conditional_losses_9445812\3?0
)?&
 ?
inputs?????????d
p 
? "%?"
?
0?????????d
? 
,__inference_dropout_47_layer_call_fn_9445817O3?0
)?&
 ?
inputs?????????d
p
? "??????????d
,__inference_dropout_47_layer_call_fn_9445822O3?0
)?&
 ?
inputs?????????d
p 
? "??????????d?
G__inference_dropout_48_layer_call_and_return_conditional_losses_9445856\3?0
)?&
 ?
inputs?????????d
p
? "%?"
?
0?????????d
? ?
G__inference_dropout_48_layer_call_and_return_conditional_losses_9445861\3?0
)?&
 ?
inputs?????????d
p 
? "%?"
?
0?????????d
? 
,__inference_dropout_48_layer_call_fn_9445866O3?0
)?&
 ?
inputs?????????d
p
? "??????????d
,__inference_dropout_48_layer_call_fn_9445871O3?0
)?&
 ?
inputs?????????d
p 
? "??????????d?
F__inference_initModel_layer_call_and_return_conditional_losses_9445325j()237?4
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
F__inference_initModel_layer_call_and_return_conditional_losses_9445363j()237?4
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
F__inference_initModel_layer_call_and_return_conditional_losses_9445596k()238?5
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
F__inference_initModel_layer_call_and_return_conditional_losses_9445641k()238?5
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
+__inference_initModel_layer_call_fn_9445423]()237?4
-?*
 ?
Input??????????
p

 
? "??????????
?
+__inference_initModel_layer_call_fn_9445482]()237?4
-?*
 ?
Input??????????
p 

 
? "??????????
?
+__inference_initModel_layer_call_fn_9445662^()238?5
.?+
!?
inputs??????????
p

 
? "??????????
?
+__inference_initModel_layer_call_fn_9445683^()238?5
.?+
!?
inputs??????????
p 

 
? "??????????
<
__inference_loss_fn_0_9445903?

? 
? "? 9
__inference_loss_fn_1_9445908?

? 
? "? 9
__inference_loss_fn_2_9445913?

? 
? "? ?
%__inference_signature_wrapper_9445523()238?5
? 
.?+
)
Input ?
Input??????????"9?6
4
Logit_Probs%?"
Logit_Probs?????????
