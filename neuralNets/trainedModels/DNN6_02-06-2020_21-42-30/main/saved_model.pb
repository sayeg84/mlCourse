Ûá
©ý
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
dtypetype
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
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.2.0-dlenv2v2.2.0-0-g2b96f368á

Dense_layer_1_55/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
È*(
shared_nameDense_layer_1_55/kernel

+Dense_layer_1_55/kernel/Read/ReadVariableOpReadVariableOpDense_layer_1_55/kernel* 
_output_shapes
:
È*
dtype0

Dense_layer_1_55/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*&
shared_nameDense_layer_1_55/bias
|
)Dense_layer_1_55/bias/Read/ReadVariableOpReadVariableOpDense_layer_1_55/bias*
_output_shapes	
:È*
dtype0

Dense_layer_2_44/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
È*(
shared_nameDense_layer_2_44/kernel

+Dense_layer_2_44/kernel/Read/ReadVariableOpReadVariableOpDense_layer_2_44/kernel* 
_output_shapes
:
È*
dtype0

Dense_layer_2_44/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameDense_layer_2_44/bias
|
)Dense_layer_2_44/bias/Read/ReadVariableOpReadVariableOpDense_layer_2_44/bias*
_output_shapes	
:*
dtype0

Dense_layer_3_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*(
shared_nameDense_layer_3_22/kernel

+Dense_layer_3_22/kernel/Read/ReadVariableOpReadVariableOpDense_layer_3_22/kernel*
_output_shapes
:	d*
dtype0

Dense_layer_3_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameDense_layer_3_22/bias
{
)Dense_layer_3_22/bias/Read/ReadVariableOpReadVariableOpDense_layer_3_22/bias*
_output_shapes
:d*
dtype0

Dense_layer_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*%
shared_nameDense_layer_4/kernel
}
(Dense_layer_4/kernel/Read/ReadVariableOpReadVariableOpDense_layer_4/kernel*
_output_shapes

:dd*
dtype0
|
Dense_layer_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*#
shared_nameDense_layer_4/bias
u
&Dense_layer_4/bias/Read/ReadVariableOpReadVariableOpDense_layer_4/bias*
_output_shapes
:d*
dtype0

Logit_Probs_55/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*&
shared_nameLogit_Probs_55/kernel

)Logit_Probs_55/kernel/Read/ReadVariableOpReadVariableOpLogit_Probs_55/kernel*
_output_shapes

:d
*
dtype0
~
Logit_Probs_55/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameLogit_Probs_55/bias
w
'Logit_Probs_55/bias/Read/ReadVariableOpReadVariableOpLogit_Probs_55/bias*
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

Adam/Dense_layer_1_55/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
È*/
shared_name Adam/Dense_layer_1_55/kernel/m

2Adam/Dense_layer_1_55/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_1_55/kernel/m* 
_output_shapes
:
È*
dtype0

Adam/Dense_layer_1_55/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*-
shared_nameAdam/Dense_layer_1_55/bias/m

0Adam/Dense_layer_1_55/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_1_55/bias/m*
_output_shapes	
:È*
dtype0

Adam/Dense_layer_2_44/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
È*/
shared_name Adam/Dense_layer_2_44/kernel/m

2Adam/Dense_layer_2_44/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_2_44/kernel/m* 
_output_shapes
:
È*
dtype0

Adam/Dense_layer_2_44/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/Dense_layer_2_44/bias/m

0Adam/Dense_layer_2_44/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_2_44/bias/m*
_output_shapes	
:*
dtype0

Adam/Dense_layer_3_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*/
shared_name Adam/Dense_layer_3_22/kernel/m

2Adam/Dense_layer_3_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_3_22/kernel/m*
_output_shapes
:	d*
dtype0

Adam/Dense_layer_3_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*-
shared_nameAdam/Dense_layer_3_22/bias/m

0Adam/Dense_layer_3_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_3_22/bias/m*
_output_shapes
:d*
dtype0

Adam/Dense_layer_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*,
shared_nameAdam/Dense_layer_4/kernel/m

/Adam/Dense_layer_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_4/kernel/m*
_output_shapes

:dd*
dtype0

Adam/Dense_layer_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d**
shared_nameAdam/Dense_layer_4/bias/m

-Adam/Dense_layer_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_4/bias/m*
_output_shapes
:d*
dtype0

Adam/Logit_Probs_55/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*-
shared_nameAdam/Logit_Probs_55/kernel/m

0Adam/Logit_Probs_55/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Logit_Probs_55/kernel/m*
_output_shapes

:d
*
dtype0

Adam/Logit_Probs_55/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_nameAdam/Logit_Probs_55/bias/m

.Adam/Logit_Probs_55/bias/m/Read/ReadVariableOpReadVariableOpAdam/Logit_Probs_55/bias/m*
_output_shapes
:
*
dtype0

Adam/Dense_layer_1_55/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
È*/
shared_name Adam/Dense_layer_1_55/kernel/v

2Adam/Dense_layer_1_55/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_1_55/kernel/v* 
_output_shapes
:
È*
dtype0

Adam/Dense_layer_1_55/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*-
shared_nameAdam/Dense_layer_1_55/bias/v

0Adam/Dense_layer_1_55/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_1_55/bias/v*
_output_shapes	
:È*
dtype0

Adam/Dense_layer_2_44/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
È*/
shared_name Adam/Dense_layer_2_44/kernel/v

2Adam/Dense_layer_2_44/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_2_44/kernel/v* 
_output_shapes
:
È*
dtype0

Adam/Dense_layer_2_44/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/Dense_layer_2_44/bias/v

0Adam/Dense_layer_2_44/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_2_44/bias/v*
_output_shapes	
:*
dtype0

Adam/Dense_layer_3_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*/
shared_name Adam/Dense_layer_3_22/kernel/v

2Adam/Dense_layer_3_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_3_22/kernel/v*
_output_shapes
:	d*
dtype0

Adam/Dense_layer_3_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*-
shared_nameAdam/Dense_layer_3_22/bias/v

0Adam/Dense_layer_3_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_3_22/bias/v*
_output_shapes
:d*
dtype0

Adam/Dense_layer_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*,
shared_nameAdam/Dense_layer_4/kernel/v

/Adam/Dense_layer_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_4/kernel/v*
_output_shapes

:dd*
dtype0

Adam/Dense_layer_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d**
shared_nameAdam/Dense_layer_4/bias/v

-Adam/Dense_layer_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_layer_4/bias/v*
_output_shapes
:d*
dtype0

Adam/Logit_Probs_55/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*-
shared_nameAdam/Logit_Probs_55/kernel/v

0Adam/Logit_Probs_55/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Logit_Probs_55/kernel/v*
_output_shapes

:d
*
dtype0

Adam/Logit_Probs_55/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_nameAdam/Logit_Probs_55/bias/v

.Adam/Logit_Probs_55/bias/v/Read/ReadVariableOpReadVariableOpAdam/Logit_Probs_55/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
øF
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*³F
value©FB¦F BF

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
layer-9
layer_with_weights-4
layer-10
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
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
R
regularization_losses
trainable_variables
	variables
	keras_api
h

 kernel
!bias
"regularization_losses
#trainable_variables
$	variables
%	keras_api
R
&regularization_losses
'trainable_variables
(	variables
)	keras_api
h

*kernel
+bias
,regularization_losses
-trainable_variables
.	variables
/	keras_api
R
0regularization_losses
1trainable_variables
2	variables
3	keras_api
h

4kernel
5bias
6regularization_losses
7trainable_variables
8	variables
9	keras_api
R
:regularization_losses
;trainable_variables
<	variables
=	keras_api
h

>kernel
?bias
@regularization_losses
Atrainable_variables
B	variables
C	keras_api

Diter

Ebeta_1

Fbeta_2
	Gdecay
Hlearning_ratemm m!m*m+m4m5m>m?mvv v!v*v+v 4v¡5v¢>v£?v¤
 
F
0
1
 2
!3
*4
+5
46
57
>8
?9
F
0
1
 2
!3
*4
+5
46
57
>8
?9
­
regularization_losses

Ilayers
trainable_variables
	variables
Jlayer_regularization_losses
Klayer_metrics
Lmetrics
Mnon_trainable_variables
 
 
 
 
­
regularization_losses

Nlayers
Olayer_regularization_losses
trainable_variables
	variables
Player_metrics
Qmetrics
Rnon_trainable_variables
ca
VARIABLE_VALUEDense_layer_1_55/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEDense_layer_1_55/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses

Slayers
Tlayer_regularization_losses
trainable_variables
	variables
Ulayer_metrics
Vmetrics
Wnon_trainable_variables
 
 
 
­
regularization_losses

Xlayers
Ylayer_regularization_losses
trainable_variables
	variables
Zlayer_metrics
[metrics
\non_trainable_variables
ca
VARIABLE_VALUEDense_layer_2_44/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEDense_layer_2_44/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1

 0
!1
­
"regularization_losses

]layers
^layer_regularization_losses
#trainable_variables
$	variables
_layer_metrics
`metrics
anon_trainable_variables
 
 
 
­
&regularization_losses

blayers
clayer_regularization_losses
'trainable_variables
(	variables
dlayer_metrics
emetrics
fnon_trainable_variables
ca
VARIABLE_VALUEDense_layer_3_22/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEDense_layer_3_22/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

*0
+1

*0
+1
­
,regularization_losses

glayers
hlayer_regularization_losses
-trainable_variables
.	variables
ilayer_metrics
jmetrics
knon_trainable_variables
 
 
 
­
0regularization_losses

llayers
mlayer_regularization_losses
1trainable_variables
2	variables
nlayer_metrics
ometrics
pnon_trainable_variables
`^
VARIABLE_VALUEDense_layer_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEDense_layer_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51

40
51
­
6regularization_losses

qlayers
rlayer_regularization_losses
7trainable_variables
8	variables
slayer_metrics
tmetrics
unon_trainable_variables
 
 
 
­
:regularization_losses

vlayers
wlayer_regularization_losses
;trainable_variables
<	variables
xlayer_metrics
ymetrics
znon_trainable_variables
a_
VARIABLE_VALUELogit_Probs_55/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUELogit_Probs_55/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

>0
?1

>0
?1
­
@regularization_losses

{layers
|layer_regularization_losses
Atrainable_variables
B	variables
}layer_metrics
~metrics
non_trainable_variables
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

0
1
2
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
8

total

count
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables

VARIABLE_VALUEAdam/Dense_layer_1_55/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/Dense_layer_1_55/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/Dense_layer_2_44/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/Dense_layer_2_44/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/Dense_layer_3_22/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/Dense_layer_3_22/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/Dense_layer_4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/Dense_layer_4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/Logit_Probs_55/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/Logit_Probs_55/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/Dense_layer_1_55/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/Dense_layer_1_55/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/Dense_layer_2_44/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/Dense_layer_2_44/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/Dense_layer_3_22/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/Dense_layer_3_22/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/Dense_layer_4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/Dense_layer_4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/Logit_Probs_55/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/Logit_Probs_55/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_InputPlaceholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_InputDense_layer_1_55/kernelDense_layer_1_55/biasDense_layer_2_44/kernelDense_layer_2_44/biasDense_layer_3_22/kernelDense_layer_3_22/biasDense_layer_4/kernelDense_layer_4/biasLogit_Probs_55/kernelLogit_Probs_55/bias*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

GPU

CPU2*0J 8*/
f*R(
&__inference_signature_wrapper_14498127
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+Dense_layer_1_55/kernel/Read/ReadVariableOp)Dense_layer_1_55/bias/Read/ReadVariableOp+Dense_layer_2_44/kernel/Read/ReadVariableOp)Dense_layer_2_44/bias/Read/ReadVariableOp+Dense_layer_3_22/kernel/Read/ReadVariableOp)Dense_layer_3_22/bias/Read/ReadVariableOp(Dense_layer_4/kernel/Read/ReadVariableOp&Dense_layer_4/bias/Read/ReadVariableOp)Logit_Probs_55/kernel/Read/ReadVariableOp'Logit_Probs_55/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp2Adam/Dense_layer_1_55/kernel/m/Read/ReadVariableOp0Adam/Dense_layer_1_55/bias/m/Read/ReadVariableOp2Adam/Dense_layer_2_44/kernel/m/Read/ReadVariableOp0Adam/Dense_layer_2_44/bias/m/Read/ReadVariableOp2Adam/Dense_layer_3_22/kernel/m/Read/ReadVariableOp0Adam/Dense_layer_3_22/bias/m/Read/ReadVariableOp/Adam/Dense_layer_4/kernel/m/Read/ReadVariableOp-Adam/Dense_layer_4/bias/m/Read/ReadVariableOp0Adam/Logit_Probs_55/kernel/m/Read/ReadVariableOp.Adam/Logit_Probs_55/bias/m/Read/ReadVariableOp2Adam/Dense_layer_1_55/kernel/v/Read/ReadVariableOp0Adam/Dense_layer_1_55/bias/v/Read/ReadVariableOp2Adam/Dense_layer_2_44/kernel/v/Read/ReadVariableOp0Adam/Dense_layer_2_44/bias/v/Read/ReadVariableOp2Adam/Dense_layer_3_22/kernel/v/Read/ReadVariableOp0Adam/Dense_layer_3_22/bias/v/Read/ReadVariableOp/Adam/Dense_layer_4/kernel/v/Read/ReadVariableOp-Adam/Dense_layer_4/bias/v/Read/ReadVariableOp0Adam/Logit_Probs_55/kernel/v/Read/ReadVariableOp.Adam/Logit_Probs_55/bias/v/Read/ReadVariableOpConst*6
Tin/
-2+	*
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
CPU2*0J 8**
f%R#
!__inference__traced_save_14498784
×	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameDense_layer_1_55/kernelDense_layer_1_55/biasDense_layer_2_44/kernelDense_layer_2_44/biasDense_layer_3_22/kernelDense_layer_3_22/biasDense_layer_4/kernelDense_layer_4/biasLogit_Probs_55/kernelLogit_Probs_55/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/Dense_layer_1_55/kernel/mAdam/Dense_layer_1_55/bias/mAdam/Dense_layer_2_44/kernel/mAdam/Dense_layer_2_44/bias/mAdam/Dense_layer_3_22/kernel/mAdam/Dense_layer_3_22/bias/mAdam/Dense_layer_4/kernel/mAdam/Dense_layer_4/bias/mAdam/Logit_Probs_55/kernel/mAdam/Logit_Probs_55/bias/mAdam/Dense_layer_1_55/kernel/vAdam/Dense_layer_1_55/bias/vAdam/Dense_layer_2_44/kernel/vAdam/Dense_layer_2_44/bias/vAdam/Dense_layer_3_22/kernel/vAdam/Dense_layer_3_22/bias/vAdam/Dense_layer_4/kernel/vAdam/Dense_layer_4/bias/vAdam/Logit_Probs_55/kernel/vAdam/Logit_Probs_55/bias/v*5
Tin.
,2**
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
CPU2*0J 8*-
f(R&
$__inference__traced_restore_14498919ÅÜ	

±
I__inference_Logit_Probs_layer_call_and_return_conditional_losses_14497867

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 


0__inference_Dense_layer_3_layer_call_fn_14498506

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_Dense_layer_3_layer_call_and_return_conditional_losses_144977482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

h
I__inference_dropout_125_layer_call_and_return_conditional_losses_14498572

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÀ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ç<
À
G__inference_initModel_layer_call_and_return_conditional_losses_14498058

inputs
dense_layer_1_14498017
dense_layer_1_14498019
dense_layer_2_14498023
dense_layer_2_14498025
dense_layer_3_14498029
dense_layer_3_14498031
dense_layer_4_14498035
dense_layer_4_14498037
logit_probs_14498041
logit_probs_14498043
identity¢%Dense_layer_1/StatefulPartitionedCall¢%Dense_layer_2/StatefulPartitionedCall¢%Dense_layer_3/StatefulPartitionedCall¢%Dense_layer_4/StatefulPartitionedCall¢#Logit_Probs/StatefulPartitionedCallÁ
dropout_121/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_121_layer_call_and_return_conditional_losses_144975952
dropout_121/PartitionedCall³
%Dense_layer_1/StatefulPartitionedCallStatefulPartitionedCall$dropout_121/PartitionedCall:output:0dense_layer_1_14498017dense_layer_1_14498019*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_Dense_layer_1_layer_call_and_return_conditional_losses_144976272'
%Dense_layer_1/StatefulPartitionedCallé
dropout_122/PartitionedCallPartitionedCall.Dense_layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_122_layer_call_and_return_conditional_losses_144976602
dropout_122/PartitionedCall³
%Dense_layer_2/StatefulPartitionedCallStatefulPartitionedCall$dropout_122/PartitionedCall:output:0dense_layer_2_14498023dense_layer_2_14498025*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_Dense_layer_2_layer_call_and_return_conditional_losses_144976902'
%Dense_layer_2/StatefulPartitionedCallé
dropout_123/PartitionedCallPartitionedCall.Dense_layer_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_123_layer_call_and_return_conditional_losses_144977232
dropout_123/PartitionedCall²
%Dense_layer_3/StatefulPartitionedCallStatefulPartitionedCall$dropout_123/PartitionedCall:output:0dense_layer_3_14498029dense_layer_3_14498031*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_Dense_layer_3_layer_call_and_return_conditional_losses_144977482'
%Dense_layer_3/StatefulPartitionedCallè
dropout_124/PartitionedCallPartitionedCall.Dense_layer_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_124_layer_call_and_return_conditional_losses_144977812
dropout_124/PartitionedCall²
%Dense_layer_4/StatefulPartitionedCallStatefulPartitionedCall$dropout_124/PartitionedCall:output:0dense_layer_4_14498035dense_layer_4_14498037*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_Dense_layer_4_layer_call_and_return_conditional_losses_144978112'
%Dense_layer_4/StatefulPartitionedCallè
dropout_125/PartitionedCallPartitionedCall.Dense_layer_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_125_layer_call_and_return_conditional_losses_144978442
dropout_125/PartitionedCall¨
#Logit_Probs/StatefulPartitionedCallStatefulPartitionedCall$dropout_125/PartitionedCall:output:0logit_probs_14498041logit_probs_14498043*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_Logit_Probs_layer_call_and_return_conditional_losses_144978672%
#Logit_Probs/StatefulPartitionedCallÉ
6Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_layer_1_14498017* 
_output_shapes
:
È*
dtype028
6Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOpÄ
'Dense_layer_1_55/kernel/Regularizer/AbsAbs>Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
È2)
'Dense_layer_1_55/kernel/Regularizer/Abs§
)Dense_layer_1_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)Dense_layer_1_55/kernel/Regularizer/ConstÛ
'Dense_layer_1_55/kernel/Regularizer/SumSum+Dense_layer_1_55/kernel/Regularizer/Abs:y:02Dense_layer_1_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/Sum
)Dense_layer_1_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2+
)Dense_layer_1_55/kernel/Regularizer/mul/xà
'Dense_layer_1_55/kernel/Regularizer/mulMul2Dense_layer_1_55/kernel/Regularizer/mul/x:output:00Dense_layer_1_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/mul
)Dense_layer_1_55/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_1_55/kernel/Regularizer/add/xÝ
'Dense_layer_1_55/kernel/Regularizer/addAddV22Dense_layer_1_55/kernel/Regularizer/add/x:output:0+Dense_layer_1_55/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/add
)Dense_layer_2_44/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_2_44/kernel/Regularizer/Const
)Dense_layer_3_22/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_3_22/kernel/Regularizer/Const
&Dense_layer_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&Dense_layer_4/kernel/Regularizer/ConstÆ
IdentityIdentity,Logit_Probs/StatefulPartitionedCall:output:0&^Dense_layer_1/StatefulPartitionedCall&^Dense_layer_2/StatefulPartitionedCall&^Dense_layer_3/StatefulPartitionedCall&^Dense_layer_4/StatefulPartitionedCall$^Logit_Probs/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿ::::::::::2N
%Dense_layer_1/StatefulPartitionedCall%Dense_layer_1/StatefulPartitionedCall2N
%Dense_layer_2/StatefulPartitionedCall%Dense_layer_2/StatefulPartitionedCall2N
%Dense_layer_3/StatefulPartitionedCall%Dense_layer_3/StatefulPartitionedCall2N
%Dense_layer_4/StatefulPartitionedCall%Dense_layer_4/StatefulPartitionedCall2J
#Logit_Probs/StatefulPartitionedCall#Logit_Probs/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
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
: :	

_output_shapes
: :


_output_shapes
: 
Ì
g
I__inference_dropout_124_layer_call_and_return_conditional_losses_14497781

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs

g
.__inference_dropout_123_layer_call_fn_14498479

inputs
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_123_layer_call_and_return_conditional_losses_144977182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
g
I__inference_dropout_122_layer_call_and_return_conditional_losses_14497660

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
ý
J
.__inference_dropout_125_layer_call_fn_14498587

inputs
identity¨
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_125_layer_call_and_return_conditional_losses_144978442
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
´
à
$__inference__traced_restore_14498919
file_prefix,
(assignvariableop_dense_layer_1_55_kernel,
(assignvariableop_1_dense_layer_1_55_bias.
*assignvariableop_2_dense_layer_2_44_kernel,
(assignvariableop_3_dense_layer_2_44_bias.
*assignvariableop_4_dense_layer_3_22_kernel,
(assignvariableop_5_dense_layer_3_22_bias+
'assignvariableop_6_dense_layer_4_kernel)
%assignvariableop_7_dense_layer_4_bias,
(assignvariableop_8_logit_probs_55_kernel*
&assignvariableop_9_logit_probs_55_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate
assignvariableop_15_total
assignvariableop_16_count
assignvariableop_17_total_1
assignvariableop_18_count_1
assignvariableop_19_total_2
assignvariableop_20_count_26
2assignvariableop_21_adam_dense_layer_1_55_kernel_m4
0assignvariableop_22_adam_dense_layer_1_55_bias_m6
2assignvariableop_23_adam_dense_layer_2_44_kernel_m4
0assignvariableop_24_adam_dense_layer_2_44_bias_m6
2assignvariableop_25_adam_dense_layer_3_22_kernel_m4
0assignvariableop_26_adam_dense_layer_3_22_bias_m3
/assignvariableop_27_adam_dense_layer_4_kernel_m1
-assignvariableop_28_adam_dense_layer_4_bias_m4
0assignvariableop_29_adam_logit_probs_55_kernel_m2
.assignvariableop_30_adam_logit_probs_55_bias_m6
2assignvariableop_31_adam_dense_layer_1_55_kernel_v4
0assignvariableop_32_adam_dense_layer_1_55_bias_v6
2assignvariableop_33_adam_dense_layer_2_44_kernel_v4
0assignvariableop_34_adam_dense_layer_2_44_bias_v6
2assignvariableop_35_adam_dense_layer_3_22_kernel_v4
0assignvariableop_36_adam_dense_layer_3_22_bias_v3
/assignvariableop_37_adam_dense_layer_4_kernel_v1
-assignvariableop_38_adam_dense_layer_4_bias_v4
0assignvariableop_39_adam_logit_probs_55_kernel_v2
.assignvariableop_40_adam_logit_probs_55_bias_v
identity_42¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¢	RestoreV2¢RestoreV2_1Ô
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*à
valueÖBÓ)B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesà
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesû
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*º
_output_shapes§
¤:::::::::::::::::::::::::::::::::::::::::*7
dtypes-
+2)	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp(assignvariableop_dense_layer_1_55_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOp(assignvariableop_1_dense_layer_1_55_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2 
AssignVariableOp_2AssignVariableOp*assignvariableop_2_dense_layer_2_44_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOp(assignvariableop_3_dense_layer_2_44_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4 
AssignVariableOp_4AssignVariableOp*assignvariableop_4_dense_layer_3_22_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOp(assignvariableop_5_dense_layer_3_22_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOp'assignvariableop_6_dense_layer_4_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOp%assignvariableop_7_dense_layer_4_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOp(assignvariableop_8_logit_probs_55_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOp&assignvariableop_9_logit_probs_55_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0	*
_output_shapes
:2
Identity_10
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_2Identity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_2Identity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21«
AssignVariableOp_21AssignVariableOp2assignvariableop_21_adam_dense_layer_1_55_kernel_mIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22©
AssignVariableOp_22AssignVariableOp0assignvariableop_22_adam_dense_layer_1_55_bias_mIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23«
AssignVariableOp_23AssignVariableOp2assignvariableop_23_adam_dense_layer_2_44_kernel_mIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24©
AssignVariableOp_24AssignVariableOp0assignvariableop_24_adam_dense_layer_2_44_bias_mIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25«
AssignVariableOp_25AssignVariableOp2assignvariableop_25_adam_dense_layer_3_22_kernel_mIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26©
AssignVariableOp_26AssignVariableOp0assignvariableop_26_adam_dense_layer_3_22_bias_mIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27¨
AssignVariableOp_27AssignVariableOp/assignvariableop_27_adam_dense_layer_4_kernel_mIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28¦
AssignVariableOp_28AssignVariableOp-assignvariableop_28_adam_dense_layer_4_bias_mIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29©
AssignVariableOp_29AssignVariableOp0assignvariableop_29_adam_logit_probs_55_kernel_mIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30§
AssignVariableOp_30AssignVariableOp.assignvariableop_30_adam_logit_probs_55_bias_mIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31«
AssignVariableOp_31AssignVariableOp2assignvariableop_31_adam_dense_layer_1_55_kernel_vIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32©
AssignVariableOp_32AssignVariableOp0assignvariableop_32_adam_dense_layer_1_55_bias_vIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33«
AssignVariableOp_33AssignVariableOp2assignvariableop_33_adam_dense_layer_2_44_kernel_vIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34©
AssignVariableOp_34AssignVariableOp0assignvariableop_34_adam_dense_layer_2_44_bias_vIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35«
AssignVariableOp_35AssignVariableOp2assignvariableop_35_adam_dense_layer_3_22_kernel_vIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36©
AssignVariableOp_36AssignVariableOp0assignvariableop_36_adam_dense_layer_3_22_bias_vIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37¨
AssignVariableOp_37AssignVariableOp/assignvariableop_37_adam_dense_layer_4_kernel_vIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38¦
AssignVariableOp_38AssignVariableOp-assignvariableop_38_adam_dense_layer_4_bias_vIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39©
AssignVariableOp_39AssignVariableOp0assignvariableop_39_adam_logit_probs_55_kernel_vIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40§
AssignVariableOp_40AssignVariableOp.assignvariableop_40_adam_logit_probs_55_bias_vIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40¨
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesÄ
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
NoOpä
Identity_41Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_41ñ
Identity_42IdentityIdentity_41:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_42"#
identity_42Identity_42:output:0*»
_input_shapes©
¦: :::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_40AssignVariableOp_402(
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
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: 

h
I__inference_dropout_121_layer_call_and_return_conditional_losses_14497590

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÁ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
.
__inference_loss_fn_2_14498629
identity
)Dense_layer_3_22/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_3_22/kernel/Regularizer/Constu
IdentityIdentity2Dense_layer_3_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
E
þ
G__inference_initModel_layer_call_and_return_conditional_losses_14497988

inputs
dense_layer_1_14497947
dense_layer_1_14497949
dense_layer_2_14497953
dense_layer_2_14497955
dense_layer_3_14497959
dense_layer_3_14497961
dense_layer_4_14497965
dense_layer_4_14497967
logit_probs_14497971
logit_probs_14497973
identity¢%Dense_layer_1/StatefulPartitionedCall¢%Dense_layer_2/StatefulPartitionedCall¢%Dense_layer_3/StatefulPartitionedCall¢%Dense_layer_4/StatefulPartitionedCall¢#Logit_Probs/StatefulPartitionedCall¢#dropout_121/StatefulPartitionedCall¢#dropout_122/StatefulPartitionedCall¢#dropout_123/StatefulPartitionedCall¢#dropout_124/StatefulPartitionedCall¢#dropout_125/StatefulPartitionedCallÙ
#dropout_121/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_121_layer_call_and_return_conditional_losses_144975902%
#dropout_121/StatefulPartitionedCall»
%Dense_layer_1/StatefulPartitionedCallStatefulPartitionedCall,dropout_121/StatefulPartitionedCall:output:0dense_layer_1_14497947dense_layer_1_14497949*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_Dense_layer_1_layer_call_and_return_conditional_losses_144976272'
%Dense_layer_1/StatefulPartitionedCall§
#dropout_122/StatefulPartitionedCallStatefulPartitionedCall.Dense_layer_1/StatefulPartitionedCall:output:0$^dropout_121/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_122_layer_call_and_return_conditional_losses_144976552%
#dropout_122/StatefulPartitionedCall»
%Dense_layer_2/StatefulPartitionedCallStatefulPartitionedCall,dropout_122/StatefulPartitionedCall:output:0dense_layer_2_14497953dense_layer_2_14497955*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_Dense_layer_2_layer_call_and_return_conditional_losses_144976902'
%Dense_layer_2/StatefulPartitionedCall§
#dropout_123/StatefulPartitionedCallStatefulPartitionedCall.Dense_layer_2/StatefulPartitionedCall:output:0$^dropout_122/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_123_layer_call_and_return_conditional_losses_144977182%
#dropout_123/StatefulPartitionedCallº
%Dense_layer_3/StatefulPartitionedCallStatefulPartitionedCall,dropout_123/StatefulPartitionedCall:output:0dense_layer_3_14497959dense_layer_3_14497961*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_Dense_layer_3_layer_call_and_return_conditional_losses_144977482'
%Dense_layer_3/StatefulPartitionedCall¦
#dropout_124/StatefulPartitionedCallStatefulPartitionedCall.Dense_layer_3/StatefulPartitionedCall:output:0$^dropout_123/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_124_layer_call_and_return_conditional_losses_144977762%
#dropout_124/StatefulPartitionedCallº
%Dense_layer_4/StatefulPartitionedCallStatefulPartitionedCall,dropout_124/StatefulPartitionedCall:output:0dense_layer_4_14497965dense_layer_4_14497967*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_Dense_layer_4_layer_call_and_return_conditional_losses_144978112'
%Dense_layer_4/StatefulPartitionedCall¦
#dropout_125/StatefulPartitionedCallStatefulPartitionedCall.Dense_layer_4/StatefulPartitionedCall:output:0$^dropout_124/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_125_layer_call_and_return_conditional_losses_144978392%
#dropout_125/StatefulPartitionedCall°
#Logit_Probs/StatefulPartitionedCallStatefulPartitionedCall,dropout_125/StatefulPartitionedCall:output:0logit_probs_14497971logit_probs_14497973*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_Logit_Probs_layer_call_and_return_conditional_losses_144978672%
#Logit_Probs/StatefulPartitionedCallÉ
6Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_layer_1_14497947* 
_output_shapes
:
È*
dtype028
6Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOpÄ
'Dense_layer_1_55/kernel/Regularizer/AbsAbs>Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
È2)
'Dense_layer_1_55/kernel/Regularizer/Abs§
)Dense_layer_1_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)Dense_layer_1_55/kernel/Regularizer/ConstÛ
'Dense_layer_1_55/kernel/Regularizer/SumSum+Dense_layer_1_55/kernel/Regularizer/Abs:y:02Dense_layer_1_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/Sum
)Dense_layer_1_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2+
)Dense_layer_1_55/kernel/Regularizer/mul/xà
'Dense_layer_1_55/kernel/Regularizer/mulMul2Dense_layer_1_55/kernel/Regularizer/mul/x:output:00Dense_layer_1_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/mul
)Dense_layer_1_55/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_1_55/kernel/Regularizer/add/xÝ
'Dense_layer_1_55/kernel/Regularizer/addAddV22Dense_layer_1_55/kernel/Regularizer/add/x:output:0+Dense_layer_1_55/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/add
)Dense_layer_2_44/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_2_44/kernel/Regularizer/Const
)Dense_layer_3_22/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_3_22/kernel/Regularizer/Const
&Dense_layer_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&Dense_layer_4/kernel/Regularizer/Const
IdentityIdentity,Logit_Probs/StatefulPartitionedCall:output:0&^Dense_layer_1/StatefulPartitionedCall&^Dense_layer_2/StatefulPartitionedCall&^Dense_layer_3/StatefulPartitionedCall&^Dense_layer_4/StatefulPartitionedCall$^Logit_Probs/StatefulPartitionedCall$^dropout_121/StatefulPartitionedCall$^dropout_122/StatefulPartitionedCall$^dropout_123/StatefulPartitionedCall$^dropout_124/StatefulPartitionedCall$^dropout_125/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿ::::::::::2N
%Dense_layer_1/StatefulPartitionedCall%Dense_layer_1/StatefulPartitionedCall2N
%Dense_layer_2/StatefulPartitionedCall%Dense_layer_2/StatefulPartitionedCall2N
%Dense_layer_3/StatefulPartitionedCall%Dense_layer_3/StatefulPartitionedCall2N
%Dense_layer_4/StatefulPartitionedCall%Dense_layer_4/StatefulPartitionedCall2J
#Logit_Probs/StatefulPartitionedCall#Logit_Probs/StatefulPartitionedCall2J
#dropout_121/StatefulPartitionedCall#dropout_121/StatefulPartitionedCall2J
#dropout_122/StatefulPartitionedCall#dropout_122/StatefulPartitionedCall2J
#dropout_123/StatefulPartitionedCall#dropout_123/StatefulPartitionedCall2J
#dropout_124/StatefulPartitionedCall#dropout_124/StatefulPartitionedCall2J
#dropout_125/StatefulPartitionedCall#dropout_125/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
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
: :	

_output_shapes
: :


_output_shapes
: 

g
.__inference_dropout_122_layer_call_fn_14498425

inputs
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_122_layer_call_and_return_conditional_losses_144976552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs


0__inference_Dense_layer_2_layer_call_fn_14498457

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_Dense_layer_2_layer_call_and_return_conditional_losses_144976902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

h
I__inference_dropout_124_layer_call_and_return_conditional_losses_14498518

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÀ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs


0__inference_Dense_layer_4_layer_call_fn_14498560

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_Dense_layer_4_layer_call_and_return_conditional_losses_144978112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

g
.__inference_dropout_125_layer_call_fn_14498582

inputs
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_125_layer_call_and_return_conditional_losses_144978392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs


0__inference_Dense_layer_1_layer_call_fn_14498403

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_Dense_layer_1_layer_call_and_return_conditional_losses_144976272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ð
g
I__inference_dropout_121_layer_call_and_return_conditional_losses_14498357

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
J
.__inference_dropout_124_layer_call_fn_14498533

inputs
identity¨
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_124_layer_call_and_return_conditional_losses_144977812
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs

h
I__inference_dropout_123_layer_call_and_return_conditional_losses_14498469

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÁ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

v
__inference_loss_fn_0_14498619C
?dense_layer_1_55_kernel_regularizer_abs_readvariableop_resource
identityò
6Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp?dense_layer_1_55_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
È*
dtype028
6Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOpÄ
'Dense_layer_1_55/kernel/Regularizer/AbsAbs>Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
È2)
'Dense_layer_1_55/kernel/Regularizer/Abs§
)Dense_layer_1_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)Dense_layer_1_55/kernel/Regularizer/ConstÛ
'Dense_layer_1_55/kernel/Regularizer/SumSum+Dense_layer_1_55/kernel/Regularizer/Abs:y:02Dense_layer_1_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/Sum
)Dense_layer_1_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2+
)Dense_layer_1_55/kernel/Regularizer/mul/xà
'Dense_layer_1_55/kernel/Regularizer/mulMul2Dense_layer_1_55/kernel/Regularizer/mul/x:output:00Dense_layer_1_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/mul
)Dense_layer_1_55/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_1_55/kernel/Regularizer/add/xÝ
'Dense_layer_1_55/kernel/Regularizer/addAddV22Dense_layer_1_55/kernel/Regularizer/add/x:output:0+Dense_layer_1_55/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/addn
IdentityIdentity+Dense_layer_1_55/kernel/Regularizer/add:z:0*
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

h
I__inference_dropout_123_layer_call_and_return_conditional_losses_14497718

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÁ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
g
I__inference_dropout_123_layer_call_and_return_conditional_losses_14498474

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

µ
K__inference_Dense_layer_4_layer_call_and_return_conditional_losses_14498551

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource

identity_1
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
Sigmoidb
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity·
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-14498543*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd2
	IdentityN
&Dense_layer_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&Dense_layer_4/kernel/Regularizer/Constj

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ªH
Û
G__inference_initModel_layer_call_and_return_conditional_losses_14498290

inputs0
,dense_layer_1_matmul_readvariableop_resource1
-dense_layer_1_biasadd_readvariableop_resource0
,dense_layer_2_matmul_readvariableop_resource1
-dense_layer_2_biasadd_readvariableop_resource0
,dense_layer_3_matmul_readvariableop_resource1
-dense_layer_3_biasadd_readvariableop_resource0
,dense_layer_4_matmul_readvariableop_resource1
-dense_layer_4_biasadd_readvariableop_resource.
*logit_probs_matmul_readvariableop_resource/
+logit_probs_biasadd_readvariableop_resource
identitys
dropout_121/IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_121/Identity¹
#Dense_layer_1/MatMul/ReadVariableOpReadVariableOp,dense_layer_1_matmul_readvariableop_resource* 
_output_shapes
:
È*
dtype02%
#Dense_layer_1/MatMul/ReadVariableOpµ
Dense_layer_1/MatMulMatMuldropout_121/Identity:output:0+Dense_layer_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Dense_layer_1/MatMul·
$Dense_layer_1/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype02&
$Dense_layer_1/BiasAdd/ReadVariableOpº
Dense_layer_1/BiasAddBiasAddDense_layer_1/MatMul:product:0,Dense_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Dense_layer_1/BiasAdd
Dense_layer_1/TanhTanhDense_layer_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Dense_layer_1/Tanh
dropout_122/IdentityIdentityDense_layer_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dropout_122/Identity¹
#Dense_layer_2/MatMul/ReadVariableOpReadVariableOp,dense_layer_2_matmul_readvariableop_resource* 
_output_shapes
:
È*
dtype02%
#Dense_layer_2/MatMul/ReadVariableOpµ
Dense_layer_2/MatMulMatMuldropout_122/Identity:output:0+Dense_layer_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_2/MatMul·
$Dense_layer_2/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02&
$Dense_layer_2/BiasAdd/ReadVariableOpº
Dense_layer_2/BiasAddBiasAddDense_layer_2/MatMul:product:0,Dense_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_2/BiasAdd
Dense_layer_2/SigmoidSigmoidDense_layer_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_2/Sigmoid
Dense_layer_2/mulMulDense_layer_2/BiasAdd:output:0Dense_layer_2/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_2/mul
Dense_layer_2/IdentityIdentityDense_layer_2/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_2/Identityñ
Dense_layer_2/IdentityN	IdentityNDense_layer_2/mul:z:0Dense_layer_2/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-14498244*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_2/IdentityN
dropout_123/IdentityIdentity Dense_layer_2/IdentityN:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_123/Identity¸
#Dense_layer_3/MatMul/ReadVariableOpReadVariableOp,dense_layer_3_matmul_readvariableop_resource*
_output_shapes
:	d*
dtype02%
#Dense_layer_3/MatMul/ReadVariableOp´
Dense_layer_3/MatMulMatMuldropout_123/Identity:output:0+Dense_layer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Dense_layer_3/MatMul¶
$Dense_layer_3/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02&
$Dense_layer_3/BiasAdd/ReadVariableOp¹
Dense_layer_3/BiasAddBiasAddDense_layer_3/MatMul:product:0,Dense_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Dense_layer_3/BiasAdd
Dense_layer_3/TanhTanhDense_layer_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Dense_layer_3/Tanh
dropout_124/IdentityIdentityDense_layer_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout_124/Identity·
#Dense_layer_4/MatMul/ReadVariableOpReadVariableOp,dense_layer_4_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02%
#Dense_layer_4/MatMul/ReadVariableOp´
Dense_layer_4/MatMulMatMuldropout_124/Identity:output:0+Dense_layer_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Dense_layer_4/MatMul¶
$Dense_layer_4/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02&
$Dense_layer_4/BiasAdd/ReadVariableOp¹
Dense_layer_4/BiasAddBiasAddDense_layer_4/MatMul:product:0,Dense_layer_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Dense_layer_4/BiasAdd
Dense_layer_4/SigmoidSigmoidDense_layer_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Dense_layer_4/Sigmoid
Dense_layer_4/mulMulDense_layer_4/BiasAdd:output:0Dense_layer_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Dense_layer_4/mul
Dense_layer_4/IdentityIdentityDense_layer_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Dense_layer_4/Identityï
Dense_layer_4/IdentityN	IdentityNDense_layer_4/mul:z:0Dense_layer_4/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-14498265*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd2
Dense_layer_4/IdentityN
dropout_125/IdentityIdentity Dense_layer_4/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout_125/Identity±
!Logit_Probs/MatMul/ReadVariableOpReadVariableOp*logit_probs_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02#
!Logit_Probs/MatMul/ReadVariableOp®
Logit_Probs/MatMulMatMuldropout_125/Identity:output:0)Logit_Probs/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
Logit_Probs/MatMul°
"Logit_Probs/BiasAdd/ReadVariableOpReadVariableOp+logit_probs_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02$
"Logit_Probs/BiasAdd/ReadVariableOp±
Logit_Probs/BiasAddBiasAddLogit_Probs/MatMul:product:0*Logit_Probs/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
Logit_Probs/BiasAddß
6Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp,dense_layer_1_matmul_readvariableop_resource* 
_output_shapes
:
È*
dtype028
6Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOpÄ
'Dense_layer_1_55/kernel/Regularizer/AbsAbs>Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
È2)
'Dense_layer_1_55/kernel/Regularizer/Abs§
)Dense_layer_1_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)Dense_layer_1_55/kernel/Regularizer/ConstÛ
'Dense_layer_1_55/kernel/Regularizer/SumSum+Dense_layer_1_55/kernel/Regularizer/Abs:y:02Dense_layer_1_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/Sum
)Dense_layer_1_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2+
)Dense_layer_1_55/kernel/Regularizer/mul/xà
'Dense_layer_1_55/kernel/Regularizer/mulMul2Dense_layer_1_55/kernel/Regularizer/mul/x:output:00Dense_layer_1_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/mul
)Dense_layer_1_55/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_1_55/kernel/Regularizer/add/xÝ
'Dense_layer_1_55/kernel/Regularizer/addAddV22Dense_layer_1_55/kernel/Regularizer/add/x:output:0+Dense_layer_1_55/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/add
)Dense_layer_2_44/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_2_44/kernel/Regularizer/Const
)Dense_layer_3_22/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_3_22/kernel/Regularizer/Const
&Dense_layer_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&Dense_layer_4/kernel/Regularizer/Constp
IdentityIdentityLogit_Probs/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::::::::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
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
: :	

_output_shapes
: :


_output_shapes
: 
ò

ù
,__inference_initModel_layer_call_fn_14498340

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_initModel_layer_call_and_return_conditional_losses_144980582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
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
: :	

_output_shapes
: :


_output_shapes
: 


³
K__inference_Dense_layer_3_layer_call_and_return_conditional_losses_14498497

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Tanh
)Dense_layer_3_22/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_3_22/kernel/Regularizer/Const\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ì
g
I__inference_dropout_124_layer_call_and_return_conditional_losses_14498523

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
B

#__inference__wrapped_model_14497574	
input:
6initmodel_dense_layer_1_matmul_readvariableop_resource;
7initmodel_dense_layer_1_biasadd_readvariableop_resource:
6initmodel_dense_layer_2_matmul_readvariableop_resource;
7initmodel_dense_layer_2_biasadd_readvariableop_resource:
6initmodel_dense_layer_3_matmul_readvariableop_resource;
7initmodel_dense_layer_3_biasadd_readvariableop_resource:
6initmodel_dense_layer_4_matmul_readvariableop_resource;
7initmodel_dense_layer_4_biasadd_readvariableop_resource8
4initmodel_logit_probs_matmul_readvariableop_resource9
5initmodel_logit_probs_biasadd_readvariableop_resource
identity
initModel/dropout_121/IdentityIdentityinput*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
initModel/dropout_121/Identity×
-initModel/Dense_layer_1/MatMul/ReadVariableOpReadVariableOp6initmodel_dense_layer_1_matmul_readvariableop_resource* 
_output_shapes
:
È*
dtype02/
-initModel/Dense_layer_1/MatMul/ReadVariableOpÝ
initModel/Dense_layer_1/MatMulMatMul'initModel/dropout_121/Identity:output:05initModel/Dense_layer_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2 
initModel/Dense_layer_1/MatMulÕ
.initModel/Dense_layer_1/BiasAdd/ReadVariableOpReadVariableOp7initmodel_dense_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype020
.initModel/Dense_layer_1/BiasAdd/ReadVariableOpâ
initModel/Dense_layer_1/BiasAddBiasAdd(initModel/Dense_layer_1/MatMul:product:06initModel/Dense_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
initModel/Dense_layer_1/BiasAdd¡
initModel/Dense_layer_1/TanhTanh(initModel/Dense_layer_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
initModel/Dense_layer_1/Tanh¡
initModel/dropout_122/IdentityIdentity initModel/Dense_layer_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2 
initModel/dropout_122/Identity×
-initModel/Dense_layer_2/MatMul/ReadVariableOpReadVariableOp6initmodel_dense_layer_2_matmul_readvariableop_resource* 
_output_shapes
:
È*
dtype02/
-initModel/Dense_layer_2/MatMul/ReadVariableOpÝ
initModel/Dense_layer_2/MatMulMatMul'initModel/dropout_122/Identity:output:05initModel/Dense_layer_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
initModel/Dense_layer_2/MatMulÕ
.initModel/Dense_layer_2/BiasAdd/ReadVariableOpReadVariableOp7initmodel_dense_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.initModel/Dense_layer_2/BiasAdd/ReadVariableOpâ
initModel/Dense_layer_2/BiasAddBiasAdd(initModel/Dense_layer_2/MatMul:product:06initModel/Dense_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
initModel/Dense_layer_2/BiasAddª
initModel/Dense_layer_2/SigmoidSigmoid(initModel/Dense_layer_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
initModel/Dense_layer_2/SigmoidÃ
initModel/Dense_layer_2/mulMul(initModel/Dense_layer_2/BiasAdd:output:0#initModel/Dense_layer_2/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
initModel/Dense_layer_2/mul¤
 initModel/Dense_layer_2/IdentityIdentityinitModel/Dense_layer_2/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 initModel/Dense_layer_2/Identity
!initModel/Dense_layer_2/IdentityN	IdentityNinitModel/Dense_layer_2/mul:z:0(initModel/Dense_layer_2/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-14497539*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2#
!initModel/Dense_layer_2/IdentityN«
initModel/dropout_123/IdentityIdentity*initModel/Dense_layer_2/IdentityN:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
initModel/dropout_123/IdentityÖ
-initModel/Dense_layer_3/MatMul/ReadVariableOpReadVariableOp6initmodel_dense_layer_3_matmul_readvariableop_resource*
_output_shapes
:	d*
dtype02/
-initModel/Dense_layer_3/MatMul/ReadVariableOpÜ
initModel/Dense_layer_3/MatMulMatMul'initModel/dropout_123/Identity:output:05initModel/Dense_layer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2 
initModel/Dense_layer_3/MatMulÔ
.initModel/Dense_layer_3/BiasAdd/ReadVariableOpReadVariableOp7initmodel_dense_layer_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype020
.initModel/Dense_layer_3/BiasAdd/ReadVariableOpá
initModel/Dense_layer_3/BiasAddBiasAdd(initModel/Dense_layer_3/MatMul:product:06initModel/Dense_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2!
initModel/Dense_layer_3/BiasAdd 
initModel/Dense_layer_3/TanhTanh(initModel/Dense_layer_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
initModel/Dense_layer_3/Tanh 
initModel/dropout_124/IdentityIdentity initModel/Dense_layer_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2 
initModel/dropout_124/IdentityÕ
-initModel/Dense_layer_4/MatMul/ReadVariableOpReadVariableOp6initmodel_dense_layer_4_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02/
-initModel/Dense_layer_4/MatMul/ReadVariableOpÜ
initModel/Dense_layer_4/MatMulMatMul'initModel/dropout_124/Identity:output:05initModel/Dense_layer_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2 
initModel/Dense_layer_4/MatMulÔ
.initModel/Dense_layer_4/BiasAdd/ReadVariableOpReadVariableOp7initmodel_dense_layer_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype020
.initModel/Dense_layer_4/BiasAdd/ReadVariableOpá
initModel/Dense_layer_4/BiasAddBiasAdd(initModel/Dense_layer_4/MatMul:product:06initModel/Dense_layer_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2!
initModel/Dense_layer_4/BiasAdd©
initModel/Dense_layer_4/SigmoidSigmoid(initModel/Dense_layer_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2!
initModel/Dense_layer_4/SigmoidÂ
initModel/Dense_layer_4/mulMul(initModel/Dense_layer_4/BiasAdd:output:0#initModel/Dense_layer_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
initModel/Dense_layer_4/mul£
 initModel/Dense_layer_4/IdentityIdentityinitModel/Dense_layer_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2"
 initModel/Dense_layer_4/Identity
!initModel/Dense_layer_4/IdentityN	IdentityNinitModel/Dense_layer_4/mul:z:0(initModel/Dense_layer_4/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-14497560*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd2#
!initModel/Dense_layer_4/IdentityNª
initModel/dropout_125/IdentityIdentity*initModel/Dense_layer_4/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2 
initModel/dropout_125/IdentityÏ
+initModel/Logit_Probs/MatMul/ReadVariableOpReadVariableOp4initmodel_logit_probs_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02-
+initModel/Logit_Probs/MatMul/ReadVariableOpÖ
initModel/Logit_Probs/MatMulMatMul'initModel/dropout_125/Identity:output:03initModel/Logit_Probs/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
initModel/Logit_Probs/MatMulÎ
,initModel/Logit_Probs/BiasAdd/ReadVariableOpReadVariableOp5initmodel_logit_probs_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02.
,initModel/Logit_Probs/BiasAdd/ReadVariableOpÙ
initModel/Logit_Probs/BiasAddBiasAdd&initModel/Logit_Probs/MatMul:product:04initModel/Logit_Probs/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
initModel/Logit_Probs/BiasAddz
IdentityIdentity&initModel/Logit_Probs/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::::::::O K
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

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
: :	

_output_shapes
: :


_output_shapes
: 
Ð
g
I__inference_dropout_123_layer_call_and_return_conditional_losses_14497723

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

³
K__inference_Dense_layer_1_layer_call_and_return_conditional_losses_14497627

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
È*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:È*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
TanhÑ
6Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
È*
dtype028
6Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOpÄ
'Dense_layer_1_55/kernel/Regularizer/AbsAbs>Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
È2)
'Dense_layer_1_55/kernel/Regularizer/Abs§
)Dense_layer_1_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)Dense_layer_1_55/kernel/Regularizer/ConstÛ
'Dense_layer_1_55/kernel/Regularizer/SumSum+Dense_layer_1_55/kernel/Regularizer/Abs:y:02Dense_layer_1_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/Sum
)Dense_layer_1_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2+
)Dense_layer_1_55/kernel/Regularizer/mul/xà
'Dense_layer_1_55/kernel/Regularizer/mulMul2Dense_layer_1_55/kernel/Regularizer/mul/x:output:00Dense_layer_1_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/mul
)Dense_layer_1_55/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_1_55/kernel/Regularizer/add/xÝ
'Dense_layer_1_55/kernel/Regularizer/addAddV22Dense_layer_1_55/kernel/Regularizer/add/x:output:0+Dense_layer_1_55/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/add]
IdentityIdentityTanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

±
I__inference_Logit_Probs_layer_call_and_return_conditional_losses_14498597

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
E
ý
G__inference_initModel_layer_call_and_return_conditional_losses_14497895	
input
dense_layer_1_14497638
dense_layer_1_14497640
dense_layer_2_14497701
dense_layer_2_14497703
dense_layer_3_14497759
dense_layer_3_14497761
dense_layer_4_14497822
dense_layer_4_14497824
logit_probs_14497878
logit_probs_14497880
identity¢%Dense_layer_1/StatefulPartitionedCall¢%Dense_layer_2/StatefulPartitionedCall¢%Dense_layer_3/StatefulPartitionedCall¢%Dense_layer_4/StatefulPartitionedCall¢#Logit_Probs/StatefulPartitionedCall¢#dropout_121/StatefulPartitionedCall¢#dropout_122/StatefulPartitionedCall¢#dropout_123/StatefulPartitionedCall¢#dropout_124/StatefulPartitionedCall¢#dropout_125/StatefulPartitionedCallØ
#dropout_121/StatefulPartitionedCallStatefulPartitionedCallinput*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_121_layer_call_and_return_conditional_losses_144975902%
#dropout_121/StatefulPartitionedCall»
%Dense_layer_1/StatefulPartitionedCallStatefulPartitionedCall,dropout_121/StatefulPartitionedCall:output:0dense_layer_1_14497638dense_layer_1_14497640*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_Dense_layer_1_layer_call_and_return_conditional_losses_144976272'
%Dense_layer_1/StatefulPartitionedCall§
#dropout_122/StatefulPartitionedCallStatefulPartitionedCall.Dense_layer_1/StatefulPartitionedCall:output:0$^dropout_121/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_122_layer_call_and_return_conditional_losses_144976552%
#dropout_122/StatefulPartitionedCall»
%Dense_layer_2/StatefulPartitionedCallStatefulPartitionedCall,dropout_122/StatefulPartitionedCall:output:0dense_layer_2_14497701dense_layer_2_14497703*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_Dense_layer_2_layer_call_and_return_conditional_losses_144976902'
%Dense_layer_2/StatefulPartitionedCall§
#dropout_123/StatefulPartitionedCallStatefulPartitionedCall.Dense_layer_2/StatefulPartitionedCall:output:0$^dropout_122/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_123_layer_call_and_return_conditional_losses_144977182%
#dropout_123/StatefulPartitionedCallº
%Dense_layer_3/StatefulPartitionedCallStatefulPartitionedCall,dropout_123/StatefulPartitionedCall:output:0dense_layer_3_14497759dense_layer_3_14497761*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_Dense_layer_3_layer_call_and_return_conditional_losses_144977482'
%Dense_layer_3/StatefulPartitionedCall¦
#dropout_124/StatefulPartitionedCallStatefulPartitionedCall.Dense_layer_3/StatefulPartitionedCall:output:0$^dropout_123/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_124_layer_call_and_return_conditional_losses_144977762%
#dropout_124/StatefulPartitionedCallº
%Dense_layer_4/StatefulPartitionedCallStatefulPartitionedCall,dropout_124/StatefulPartitionedCall:output:0dense_layer_4_14497822dense_layer_4_14497824*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_Dense_layer_4_layer_call_and_return_conditional_losses_144978112'
%Dense_layer_4/StatefulPartitionedCall¦
#dropout_125/StatefulPartitionedCallStatefulPartitionedCall.Dense_layer_4/StatefulPartitionedCall:output:0$^dropout_124/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_125_layer_call_and_return_conditional_losses_144978392%
#dropout_125/StatefulPartitionedCall°
#Logit_Probs/StatefulPartitionedCallStatefulPartitionedCall,dropout_125/StatefulPartitionedCall:output:0logit_probs_14497878logit_probs_14497880*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_Logit_Probs_layer_call_and_return_conditional_losses_144978672%
#Logit_Probs/StatefulPartitionedCallÉ
6Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_layer_1_14497638* 
_output_shapes
:
È*
dtype028
6Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOpÄ
'Dense_layer_1_55/kernel/Regularizer/AbsAbs>Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
È2)
'Dense_layer_1_55/kernel/Regularizer/Abs§
)Dense_layer_1_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)Dense_layer_1_55/kernel/Regularizer/ConstÛ
'Dense_layer_1_55/kernel/Regularizer/SumSum+Dense_layer_1_55/kernel/Regularizer/Abs:y:02Dense_layer_1_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/Sum
)Dense_layer_1_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2+
)Dense_layer_1_55/kernel/Regularizer/mul/xà
'Dense_layer_1_55/kernel/Regularizer/mulMul2Dense_layer_1_55/kernel/Regularizer/mul/x:output:00Dense_layer_1_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/mul
)Dense_layer_1_55/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_1_55/kernel/Regularizer/add/xÝ
'Dense_layer_1_55/kernel/Regularizer/addAddV22Dense_layer_1_55/kernel/Regularizer/add/x:output:0+Dense_layer_1_55/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/add
)Dense_layer_2_44/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_2_44/kernel/Regularizer/Const
)Dense_layer_3_22/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_3_22/kernel/Regularizer/Const
&Dense_layer_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&Dense_layer_4/kernel/Regularizer/Const
IdentityIdentity,Logit_Probs/StatefulPartitionedCall:output:0&^Dense_layer_1/StatefulPartitionedCall&^Dense_layer_2/StatefulPartitionedCall&^Dense_layer_3/StatefulPartitionedCall&^Dense_layer_4/StatefulPartitionedCall$^Logit_Probs/StatefulPartitionedCall$^dropout_121/StatefulPartitionedCall$^dropout_122/StatefulPartitionedCall$^dropout_123/StatefulPartitionedCall$^dropout_124/StatefulPartitionedCall$^dropout_125/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿ::::::::::2N
%Dense_layer_1/StatefulPartitionedCall%Dense_layer_1/StatefulPartitionedCall2N
%Dense_layer_2/StatefulPartitionedCall%Dense_layer_2/StatefulPartitionedCall2N
%Dense_layer_3/StatefulPartitionedCall%Dense_layer_3/StatefulPartitionedCall2N
%Dense_layer_4/StatefulPartitionedCall%Dense_layer_4/StatefulPartitionedCall2J
#Logit_Probs/StatefulPartitionedCall#Logit_Probs/StatefulPartitionedCall2J
#dropout_121/StatefulPartitionedCall#dropout_121/StatefulPartitionedCall2J
#dropout_122/StatefulPartitionedCall#dropout_122/StatefulPartitionedCall2J
#dropout_123/StatefulPartitionedCall#dropout_123/StatefulPartitionedCall2J
#dropout_124/StatefulPartitionedCall#dropout_124/StatefulPartitionedCall2J
#dropout_125/StatefulPartitionedCall#dropout_125/StatefulPartitionedCall:O K
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

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
: :	

_output_shapes
: :


_output_shapes
: 

g
.__inference_dropout_124_layer_call_fn_14498528

inputs
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_124_layer_call_and_return_conditional_losses_144977762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ò

ù
,__inference_initModel_layer_call_fn_14498315

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_initModel_layer_call_and_return_conditional_losses_144979882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
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
: :	

_output_shapes
: :


_output_shapes
: 

h
I__inference_dropout_125_layer_call_and_return_conditional_losses_14497839

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÀ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ï

ø
,__inference_initModel_layer_call_fn_14498011	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCall¿
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_initModel_layer_call_and_return_conditional_losses_144979882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

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
: :	

_output_shapes
: :


_output_shapes
: 

h
I__inference_dropout_121_layer_call_and_return_conditional_losses_14498352

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÁ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
g
I__inference_dropout_121_layer_call_and_return_conditional_losses_14497595

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
g
I__inference_dropout_125_layer_call_and_return_conditional_losses_14498577

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs

h
I__inference_dropout_122_layer_call_and_return_conditional_losses_14498415

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÁ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
¸y
Û
G__inference_initModel_layer_call_and_return_conditional_losses_14498226

inputs0
,dense_layer_1_matmul_readvariableop_resource1
-dense_layer_1_biasadd_readvariableop_resource0
,dense_layer_2_matmul_readvariableop_resource1
-dense_layer_2_biasadd_readvariableop_resource0
,dense_layer_3_matmul_readvariableop_resource1
-dense_layer_3_biasadd_readvariableop_resource0
,dense_layer_4_matmul_readvariableop_resource1
-dense_layer_4_biasadd_readvariableop_resource.
*logit_probs_matmul_readvariableop_resource/
+logit_probs_biasadd_readvariableop_resource
identity{
dropout_121/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_121/dropout/Const
dropout_121/dropout/MulMulinputs"dropout_121/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_121/dropout/Mull
dropout_121/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout_121/dropout/Shapeå
0dropout_121/dropout/random_uniform/RandomUniformRandomUniform"dropout_121/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed22
0dropout_121/dropout/random_uniform/RandomUniform
"dropout_121/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2$
"dropout_121/dropout/GreaterEqual/yï
 dropout_121/dropout/GreaterEqualGreaterEqual9dropout_121/dropout/random_uniform/RandomUniform:output:0+dropout_121/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 dropout_121/dropout/GreaterEqual¤
dropout_121/dropout/CastCast$dropout_121/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_121/dropout/Cast«
dropout_121/dropout/Mul_1Muldropout_121/dropout/Mul:z:0dropout_121/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_121/dropout/Mul_1¹
#Dense_layer_1/MatMul/ReadVariableOpReadVariableOp,dense_layer_1_matmul_readvariableop_resource* 
_output_shapes
:
È*
dtype02%
#Dense_layer_1/MatMul/ReadVariableOpµ
Dense_layer_1/MatMulMatMuldropout_121/dropout/Mul_1:z:0+Dense_layer_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Dense_layer_1/MatMul·
$Dense_layer_1/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype02&
$Dense_layer_1/BiasAdd/ReadVariableOpº
Dense_layer_1/BiasAddBiasAddDense_layer_1/MatMul:product:0,Dense_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Dense_layer_1/BiasAdd
Dense_layer_1/TanhTanhDense_layer_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Dense_layer_1/Tanh{
dropout_122/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout_122/dropout/Const¨
dropout_122/dropout/MulMulDense_layer_1/Tanh:y:0"dropout_122/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dropout_122/dropout/Mul|
dropout_122/dropout/ShapeShapeDense_layer_1/Tanh:y:0*
T0*
_output_shapes
:2
dropout_122/dropout/Shapeò
0dropout_122/dropout/random_uniform/RandomUniformRandomUniform"dropout_122/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0*

seed*
seed222
0dropout_122/dropout/random_uniform/RandomUniform
"dropout_122/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"dropout_122/dropout/GreaterEqual/yï
 dropout_122/dropout/GreaterEqualGreaterEqual9dropout_122/dropout/random_uniform/RandomUniform:output:0+dropout_122/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2"
 dropout_122/dropout/GreaterEqual¤
dropout_122/dropout/CastCast$dropout_122/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dropout_122/dropout/Cast«
dropout_122/dropout/Mul_1Muldropout_122/dropout/Mul:z:0dropout_122/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dropout_122/dropout/Mul_1¹
#Dense_layer_2/MatMul/ReadVariableOpReadVariableOp,dense_layer_2_matmul_readvariableop_resource* 
_output_shapes
:
È*
dtype02%
#Dense_layer_2/MatMul/ReadVariableOpµ
Dense_layer_2/MatMulMatMuldropout_122/dropout/Mul_1:z:0+Dense_layer_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_2/MatMul·
$Dense_layer_2/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02&
$Dense_layer_2/BiasAdd/ReadVariableOpº
Dense_layer_2/BiasAddBiasAddDense_layer_2/MatMul:product:0,Dense_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_2/BiasAdd
Dense_layer_2/SigmoidSigmoidDense_layer_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_2/Sigmoid
Dense_layer_2/mulMulDense_layer_2/BiasAdd:output:0Dense_layer_2/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_2/mul
Dense_layer_2/IdentityIdentityDense_layer_2/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_2/Identityñ
Dense_layer_2/IdentityN	IdentityNDense_layer_2/mul:z:0Dense_layer_2/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-14498159*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_2/IdentityN{
dropout_123/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout_123/dropout/Const²
dropout_123/dropout/MulMul Dense_layer_2/IdentityN:output:0"dropout_123/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_123/dropout/Mul
dropout_123/dropout/ShapeShape Dense_layer_2/IdentityN:output:0*
T0*
_output_shapes
:2
dropout_123/dropout/Shapeò
0dropout_123/dropout/random_uniform/RandomUniformRandomUniform"dropout_123/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed*
seed222
0dropout_123/dropout/random_uniform/RandomUniform
"dropout_123/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"dropout_123/dropout/GreaterEqual/yï
 dropout_123/dropout/GreaterEqualGreaterEqual9dropout_123/dropout/random_uniform/RandomUniform:output:0+dropout_123/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 dropout_123/dropout/GreaterEqual¤
dropout_123/dropout/CastCast$dropout_123/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_123/dropout/Cast«
dropout_123/dropout/Mul_1Muldropout_123/dropout/Mul:z:0dropout_123/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_123/dropout/Mul_1¸
#Dense_layer_3/MatMul/ReadVariableOpReadVariableOp,dense_layer_3_matmul_readvariableop_resource*
_output_shapes
:	d*
dtype02%
#Dense_layer_3/MatMul/ReadVariableOp´
Dense_layer_3/MatMulMatMuldropout_123/dropout/Mul_1:z:0+Dense_layer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Dense_layer_3/MatMul¶
$Dense_layer_3/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02&
$Dense_layer_3/BiasAdd/ReadVariableOp¹
Dense_layer_3/BiasAddBiasAddDense_layer_3/MatMul:product:0,Dense_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Dense_layer_3/BiasAdd
Dense_layer_3/TanhTanhDense_layer_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Dense_layer_3/Tanh{
dropout_124/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout_124/dropout/Const§
dropout_124/dropout/MulMulDense_layer_3/Tanh:y:0"dropout_124/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout_124/dropout/Mul|
dropout_124/dropout/ShapeShapeDense_layer_3/Tanh:y:0*
T0*
_output_shapes
:2
dropout_124/dropout/Shapeñ
0dropout_124/dropout/random_uniform/RandomUniformRandomUniform"dropout_124/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0*

seed*
seed222
0dropout_124/dropout/random_uniform/RandomUniform
"dropout_124/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"dropout_124/dropout/GreaterEqual/yî
 dropout_124/dropout/GreaterEqualGreaterEqual9dropout_124/dropout/random_uniform/RandomUniform:output:0+dropout_124/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2"
 dropout_124/dropout/GreaterEqual£
dropout_124/dropout/CastCast$dropout_124/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout_124/dropout/Castª
dropout_124/dropout/Mul_1Muldropout_124/dropout/Mul:z:0dropout_124/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout_124/dropout/Mul_1·
#Dense_layer_4/MatMul/ReadVariableOpReadVariableOp,dense_layer_4_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02%
#Dense_layer_4/MatMul/ReadVariableOp´
Dense_layer_4/MatMulMatMuldropout_124/dropout/Mul_1:z:0+Dense_layer_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Dense_layer_4/MatMul¶
$Dense_layer_4/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02&
$Dense_layer_4/BiasAdd/ReadVariableOp¹
Dense_layer_4/BiasAddBiasAddDense_layer_4/MatMul:product:0,Dense_layer_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Dense_layer_4/BiasAdd
Dense_layer_4/SigmoidSigmoidDense_layer_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Dense_layer_4/Sigmoid
Dense_layer_4/mulMulDense_layer_4/BiasAdd:output:0Dense_layer_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Dense_layer_4/mul
Dense_layer_4/IdentityIdentityDense_layer_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Dense_layer_4/Identityï
Dense_layer_4/IdentityN	IdentityNDense_layer_4/mul:z:0Dense_layer_4/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-14498194*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd2
Dense_layer_4/IdentityN{
dropout_125/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout_125/dropout/Const±
dropout_125/dropout/MulMul Dense_layer_4/IdentityN:output:0"dropout_125/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout_125/dropout/Mul
dropout_125/dropout/ShapeShape Dense_layer_4/IdentityN:output:0*
T0*
_output_shapes
:2
dropout_125/dropout/Shapeñ
0dropout_125/dropout/random_uniform/RandomUniformRandomUniform"dropout_125/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0*

seed*
seed222
0dropout_125/dropout/random_uniform/RandomUniform
"dropout_125/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"dropout_125/dropout/GreaterEqual/yî
 dropout_125/dropout/GreaterEqualGreaterEqual9dropout_125/dropout/random_uniform/RandomUniform:output:0+dropout_125/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2"
 dropout_125/dropout/GreaterEqual£
dropout_125/dropout/CastCast$dropout_125/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout_125/dropout/Castª
dropout_125/dropout/Mul_1Muldropout_125/dropout/Mul:z:0dropout_125/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout_125/dropout/Mul_1±
!Logit_Probs/MatMul/ReadVariableOpReadVariableOp*logit_probs_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02#
!Logit_Probs/MatMul/ReadVariableOp®
Logit_Probs/MatMulMatMuldropout_125/dropout/Mul_1:z:0)Logit_Probs/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
Logit_Probs/MatMul°
"Logit_Probs/BiasAdd/ReadVariableOpReadVariableOp+logit_probs_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02$
"Logit_Probs/BiasAdd/ReadVariableOp±
Logit_Probs/BiasAddBiasAddLogit_Probs/MatMul:product:0*Logit_Probs/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
Logit_Probs/BiasAddß
6Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp,dense_layer_1_matmul_readvariableop_resource* 
_output_shapes
:
È*
dtype028
6Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOpÄ
'Dense_layer_1_55/kernel/Regularizer/AbsAbs>Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
È2)
'Dense_layer_1_55/kernel/Regularizer/Abs§
)Dense_layer_1_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)Dense_layer_1_55/kernel/Regularizer/ConstÛ
'Dense_layer_1_55/kernel/Regularizer/SumSum+Dense_layer_1_55/kernel/Regularizer/Abs:y:02Dense_layer_1_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/Sum
)Dense_layer_1_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2+
)Dense_layer_1_55/kernel/Regularizer/mul/xà
'Dense_layer_1_55/kernel/Regularizer/mulMul2Dense_layer_1_55/kernel/Regularizer/mul/x:output:00Dense_layer_1_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/mul
)Dense_layer_1_55/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_1_55/kernel/Regularizer/add/xÝ
'Dense_layer_1_55/kernel/Regularizer/addAddV22Dense_layer_1_55/kernel/Regularizer/add/x:output:0+Dense_layer_1_55/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/add
)Dense_layer_2_44/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_2_44/kernel/Regularizer/Const
)Dense_layer_3_22/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_3_22/kernel/Regularizer/Const
&Dense_layer_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&Dense_layer_4/kernel/Regularizer/Constp
IdentityIdentityLogit_Probs/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::::::::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
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
: :	

_output_shapes
: :


_output_shapes
: 
Ð
g
I__inference_dropout_122_layer_call_and_return_conditional_losses_14498420

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
ï

ø
,__inference_initModel_layer_call_fn_14498081	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCall¿
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_initModel_layer_call_and_return_conditional_losses_144980582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

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
: :	

_output_shapes
: :


_output_shapes
: 
¥
µ
K__inference_Dense_layer_2_layer_call_and_return_conditional_losses_14498448

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource

identity_1
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
È*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidc
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul\
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¹
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-14498440*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2
	IdentityN
)Dense_layer_2_44/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_2_44/kernel/Regularizer/Constk

Identity_1IdentityIdentityN:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

J
.__inference_dropout_123_layer_call_fn_14498484

inputs
identity©
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_123_layer_call_and_return_conditional_losses_144977232
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
g
I__inference_dropout_125_layer_call_and_return_conditional_losses_14497844

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs

³
K__inference_Dense_layer_1_layer_call_and_return_conditional_losses_14498394

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
È*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:È*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
TanhÑ
6Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
È*
dtype028
6Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOpÄ
'Dense_layer_1_55/kernel/Regularizer/AbsAbs>Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
È2)
'Dense_layer_1_55/kernel/Regularizer/Abs§
)Dense_layer_1_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)Dense_layer_1_55/kernel/Regularizer/ConstÛ
'Dense_layer_1_55/kernel/Regularizer/SumSum+Dense_layer_1_55/kernel/Regularizer/Abs:y:02Dense_layer_1_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/Sum
)Dense_layer_1_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2+
)Dense_layer_1_55/kernel/Regularizer/mul/xà
'Dense_layer_1_55/kernel/Regularizer/mulMul2Dense_layer_1_55/kernel/Regularizer/mul/x:output:00Dense_layer_1_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/mul
)Dense_layer_1_55/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_1_55/kernel/Regularizer/add/xÝ
'Dense_layer_1_55/kernel/Regularizer/addAddV22Dense_layer_1_55/kernel/Regularizer/add/x:output:0+Dense_layer_1_55/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/add]
IdentityIdentityTanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

µ
K__inference_Dense_layer_4_layer_call_and_return_conditional_losses_14497811

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource

identity_1
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
Sigmoidb
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity·
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-14497803*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd2
	IdentityN
&Dense_layer_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&Dense_layer_4/kernel/Regularizer/Constj

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 


³
K__inference_Dense_layer_3_layer_call_and_return_conditional_losses_14497748

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Tanh
)Dense_layer_3_22/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_3_22/kernel/Regularizer/Const\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

h
I__inference_dropout_122_layer_call_and_return_conditional_losses_14497655

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÁ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs

h
I__inference_dropout_124_layer_call_and_return_conditional_losses_14497776

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÀ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs

J
.__inference_dropout_121_layer_call_fn_14498367

inputs
identity©
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_121_layer_call_and_return_conditional_losses_144975952
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥
µ
K__inference_Dense_layer_2_layer_call_and_return_conditional_losses_14497690

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource

identity_1
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
È*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidc
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul\
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¹
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-14497682*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2
	IdentityN
)Dense_layer_2_44/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_2_44/kernel/Regularizer/Constk

Identity_1IdentityIdentityN:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

J
.__inference_dropout_122_layer_call_fn_14498430

inputs
identity©
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_122_layer_call_and_return_conditional_losses_144976602
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
Å

ò
&__inference_signature_wrapper_14498127	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

GPU

CPU2*0J 8*,
f'R%
#__inference__wrapped_model_144975742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

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
: :	

_output_shapes
: :


_output_shapes
: 


.__inference_Logit_Probs_layer_call_fn_14498606

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_Logit_Probs_layer_call_and_return_conditional_losses_144978672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

g
.__inference_dropout_121_layer_call_fn_14498362

inputs
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_121_layer_call_and_return_conditional_losses_144975902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
.
__inference_loss_fn_1_14498624
identity
)Dense_layer_2_44/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_2_44/kernel/Regularizer/Constu
IdentityIdentity2Dense_layer_2_44/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
Á_
°
!__inference__traced_save_14498784
file_prefix6
2savev2_dense_layer_1_55_kernel_read_readvariableop4
0savev2_dense_layer_1_55_bias_read_readvariableop6
2savev2_dense_layer_2_44_kernel_read_readvariableop4
0savev2_dense_layer_2_44_bias_read_readvariableop6
2savev2_dense_layer_3_22_kernel_read_readvariableop4
0savev2_dense_layer_3_22_bias_read_readvariableop3
/savev2_dense_layer_4_kernel_read_readvariableop1
-savev2_dense_layer_4_bias_read_readvariableop4
0savev2_logit_probs_55_kernel_read_readvariableop2
.savev2_logit_probs_55_bias_read_readvariableop(
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
9savev2_adam_dense_layer_1_55_kernel_m_read_readvariableop;
7savev2_adam_dense_layer_1_55_bias_m_read_readvariableop=
9savev2_adam_dense_layer_2_44_kernel_m_read_readvariableop;
7savev2_adam_dense_layer_2_44_bias_m_read_readvariableop=
9savev2_adam_dense_layer_3_22_kernel_m_read_readvariableop;
7savev2_adam_dense_layer_3_22_bias_m_read_readvariableop:
6savev2_adam_dense_layer_4_kernel_m_read_readvariableop8
4savev2_adam_dense_layer_4_bias_m_read_readvariableop;
7savev2_adam_logit_probs_55_kernel_m_read_readvariableop9
5savev2_adam_logit_probs_55_bias_m_read_readvariableop=
9savev2_adam_dense_layer_1_55_kernel_v_read_readvariableop;
7savev2_adam_dense_layer_1_55_bias_v_read_readvariableop=
9savev2_adam_dense_layer_2_44_kernel_v_read_readvariableop;
7savev2_adam_dense_layer_2_44_bias_v_read_readvariableop=
9savev2_adam_dense_layer_3_22_kernel_v_read_readvariableop;
7savev2_adam_dense_layer_3_22_bias_v_read_readvariableop:
6savev2_adam_dense_layer_4_kernel_v_read_readvariableop8
4savev2_adam_dense_layer_4_bias_v_read_readvariableop;
7savev2_adam_logit_probs_55_kernel_v_read_readvariableop9
5savev2_adam_logit_probs_55_bias_v_read_readvariableop
savev2_1_const

identity_1¢MergeV2Checkpoints¢SaveV2¢SaveV2_1
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_03530f3011414393824bc8aecf66d72c/part2	
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
value	B :2

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
ShardedFilenameÎ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*à
valueÖBÓ)B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesÚ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÜ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_dense_layer_1_55_kernel_read_readvariableop0savev2_dense_layer_1_55_bias_read_readvariableop2savev2_dense_layer_2_44_kernel_read_readvariableop0savev2_dense_layer_2_44_bias_read_readvariableop2savev2_dense_layer_3_22_kernel_read_readvariableop0savev2_dense_layer_3_22_bias_read_readvariableop/savev2_dense_layer_4_kernel_read_readvariableop-savev2_dense_layer_4_bias_read_readvariableop0savev2_logit_probs_55_kernel_read_readvariableop.savev2_logit_probs_55_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop9savev2_adam_dense_layer_1_55_kernel_m_read_readvariableop7savev2_adam_dense_layer_1_55_bias_m_read_readvariableop9savev2_adam_dense_layer_2_44_kernel_m_read_readvariableop7savev2_adam_dense_layer_2_44_bias_m_read_readvariableop9savev2_adam_dense_layer_3_22_kernel_m_read_readvariableop7savev2_adam_dense_layer_3_22_bias_m_read_readvariableop6savev2_adam_dense_layer_4_kernel_m_read_readvariableop4savev2_adam_dense_layer_4_bias_m_read_readvariableop7savev2_adam_logit_probs_55_kernel_m_read_readvariableop5savev2_adam_logit_probs_55_bias_m_read_readvariableop9savev2_adam_dense_layer_1_55_kernel_v_read_readvariableop7savev2_adam_dense_layer_1_55_bias_v_read_readvariableop9savev2_adam_dense_layer_2_44_kernel_v_read_readvariableop7savev2_adam_dense_layer_2_44_bias_v_read_readvariableop9savev2_adam_dense_layer_3_22_kernel_v_read_readvariableop7savev2_adam_dense_layer_3_22_bias_v_read_readvariableop6savev2_adam_dense_layer_4_kernel_v_read_readvariableop4savev2_adam_dense_layer_4_bias_v_read_readvariableop7savev2_adam_logit_probs_55_kernel_v_read_readvariableop5savev2_adam_logit_probs_55_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *7
dtypes-
+2)	2
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard¬
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1¢
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesÏ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1ã
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¬
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*´
_input_shapes¢
: :
È:È:
È::	d:d:dd:d:d
:
: : : : : : : : : : : :
È:È:
È::	d:d:dd:d:d
:
:
È:È:
È::	d:d:dd:d:d
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
È:!

_output_shapes	
:È:&"
 
_output_shapes
:
È:!

_output_shapes	
::%!

_output_shapes
:	d: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$	 

_output_shapes

:d
: 


_output_shapes
:
:
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
: :&"
 
_output_shapes
:
È:!

_output_shapes	
:È:&"
 
_output_shapes
:
È:!

_output_shapes	
::%!

_output_shapes
:	d: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d
: 

_output_shapes
:
:& "
 
_output_shapes
:
È:!!

_output_shapes	
:È:&""
 
_output_shapes
:
È:!#

_output_shapes	
::%$!

_output_shapes
:	d: %

_output_shapes
:d:$& 

_output_shapes

:dd: '

_output_shapes
:d:$( 

_output_shapes

:d
: )

_output_shapes
:
:*

_output_shapes
: 
î
.
__inference_loss_fn_3_14498634
identity
&Dense_layer_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&Dense_layer_4/kernel/Regularizer/Constr
IdentityIdentity/Dense_layer_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
ä<
¿
G__inference_initModel_layer_call_and_return_conditional_losses_14497940	
input
dense_layer_1_14497899
dense_layer_1_14497901
dense_layer_2_14497905
dense_layer_2_14497907
dense_layer_3_14497911
dense_layer_3_14497913
dense_layer_4_14497917
dense_layer_4_14497919
logit_probs_14497923
logit_probs_14497925
identity¢%Dense_layer_1/StatefulPartitionedCall¢%Dense_layer_2/StatefulPartitionedCall¢%Dense_layer_3/StatefulPartitionedCall¢%Dense_layer_4/StatefulPartitionedCall¢#Logit_Probs/StatefulPartitionedCallÀ
dropout_121/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_121_layer_call_and_return_conditional_losses_144975952
dropout_121/PartitionedCall³
%Dense_layer_1/StatefulPartitionedCallStatefulPartitionedCall$dropout_121/PartitionedCall:output:0dense_layer_1_14497899dense_layer_1_14497901*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_Dense_layer_1_layer_call_and_return_conditional_losses_144976272'
%Dense_layer_1/StatefulPartitionedCallé
dropout_122/PartitionedCallPartitionedCall.Dense_layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_122_layer_call_and_return_conditional_losses_144976602
dropout_122/PartitionedCall³
%Dense_layer_2/StatefulPartitionedCallStatefulPartitionedCall$dropout_122/PartitionedCall:output:0dense_layer_2_14497905dense_layer_2_14497907*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_Dense_layer_2_layer_call_and_return_conditional_losses_144976902'
%Dense_layer_2/StatefulPartitionedCallé
dropout_123/PartitionedCallPartitionedCall.Dense_layer_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_123_layer_call_and_return_conditional_losses_144977232
dropout_123/PartitionedCall²
%Dense_layer_3/StatefulPartitionedCallStatefulPartitionedCall$dropout_123/PartitionedCall:output:0dense_layer_3_14497911dense_layer_3_14497913*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_Dense_layer_3_layer_call_and_return_conditional_losses_144977482'
%Dense_layer_3/StatefulPartitionedCallè
dropout_124/PartitionedCallPartitionedCall.Dense_layer_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_124_layer_call_and_return_conditional_losses_144977812
dropout_124/PartitionedCall²
%Dense_layer_4/StatefulPartitionedCallStatefulPartitionedCall$dropout_124/PartitionedCall:output:0dense_layer_4_14497917dense_layer_4_14497919*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_Dense_layer_4_layer_call_and_return_conditional_losses_144978112'
%Dense_layer_4/StatefulPartitionedCallè
dropout_125/PartitionedCallPartitionedCall.Dense_layer_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout_125_layer_call_and_return_conditional_losses_144978442
dropout_125/PartitionedCall¨
#Logit_Probs/StatefulPartitionedCallStatefulPartitionedCall$dropout_125/PartitionedCall:output:0logit_probs_14497923logit_probs_14497925*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_Logit_Probs_layer_call_and_return_conditional_losses_144978672%
#Logit_Probs/StatefulPartitionedCallÉ
6Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_layer_1_14497899* 
_output_shapes
:
È*
dtype028
6Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOpÄ
'Dense_layer_1_55/kernel/Regularizer/AbsAbs>Dense_layer_1_55/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
È2)
'Dense_layer_1_55/kernel/Regularizer/Abs§
)Dense_layer_1_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)Dense_layer_1_55/kernel/Regularizer/ConstÛ
'Dense_layer_1_55/kernel/Regularizer/SumSum+Dense_layer_1_55/kernel/Regularizer/Abs:y:02Dense_layer_1_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/Sum
)Dense_layer_1_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2+
)Dense_layer_1_55/kernel/Regularizer/mul/xà
'Dense_layer_1_55/kernel/Regularizer/mulMul2Dense_layer_1_55/kernel/Regularizer/mul/x:output:00Dense_layer_1_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/mul
)Dense_layer_1_55/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_1_55/kernel/Regularizer/add/xÝ
'Dense_layer_1_55/kernel/Regularizer/addAddV22Dense_layer_1_55/kernel/Regularizer/add/x:output:0+Dense_layer_1_55/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'Dense_layer_1_55/kernel/Regularizer/add
)Dense_layer_2_44/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_2_44/kernel/Regularizer/Const
)Dense_layer_3_22/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)Dense_layer_3_22/kernel/Regularizer/Const
&Dense_layer_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&Dense_layer_4/kernel/Regularizer/ConstÆ
IdentityIdentity,Logit_Probs/StatefulPartitionedCall:output:0&^Dense_layer_1/StatefulPartitionedCall&^Dense_layer_2/StatefulPartitionedCall&^Dense_layer_3/StatefulPartitionedCall&^Dense_layer_4/StatefulPartitionedCall$^Logit_Probs/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿ::::::::::2N
%Dense_layer_1/StatefulPartitionedCall%Dense_layer_1/StatefulPartitionedCall2N
%Dense_layer_2/StatefulPartitionedCall%Dense_layer_2/StatefulPartitionedCall2N
%Dense_layer_3/StatefulPartitionedCall%Dense_layer_3/StatefulPartitionedCall2N
%Dense_layer_4/StatefulPartitionedCall%Dense_layer_4/StatefulPartitionedCall2J
#Logit_Probs/StatefulPartitionedCall#Logit_Probs/StatefulPartitionedCall:O K
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

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
: :	

_output_shapes
: :


_output_shapes
: "¯L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*«
serving_default
8
Input/
serving_default_Input:0ÿÿÿÿÿÿÿÿÿ?
Logit_Probs0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:éÃ
M
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
layer-9
layer_with_weights-4
layer-10
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
¥_default_save_signature
+¦&call_and_return_all_conditional_losses
§__call__"¡I
_tf_keras_modelI{"class_name": "Model", "name": "initModel", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "initModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}, "name": "Input", "inbound_nodes": []}, {"class_name": "Dropout", "config": {"name": "dropout_121", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_121", "inbound_nodes": [[["Input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_layer_1", "trainable": true, "dtype": "float32", "units": 200, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0010000000474974513, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_layer_1", "inbound_nodes": [[["dropout_121", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_122", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_122", "inbound_nodes": [[["Dense_layer_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_layer_2", "trainable": true, "dtype": "float32", "units": 150, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_layer_2", "inbound_nodes": [[["dropout_122", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_123", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_123", "inbound_nodes": [[["Dense_layer_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_layer_3", "trainable": true, "dtype": "float32", "units": 100, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_layer_3", "inbound_nodes": [[["dropout_123", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_124", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_124", "inbound_nodes": [[["Dense_layer_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_layer_4", "trainable": true, "dtype": "float32", "units": 100, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_layer_4", "inbound_nodes": [[["dropout_124", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_125", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_125", "inbound_nodes": [[["Dense_layer_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Logit_Probs", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Logit_Probs", "inbound_nodes": [[["dropout_125", 0, 0, {}]]]}], "input_layers": [["Input", 0, 0]], "output_layers": [["Logit_Probs", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "initModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}, "name": "Input", "inbound_nodes": []}, {"class_name": "Dropout", "config": {"name": "dropout_121", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_121", "inbound_nodes": [[["Input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_layer_1", "trainable": true, "dtype": "float32", "units": 200, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0010000000474974513, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_layer_1", "inbound_nodes": [[["dropout_121", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_122", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_122", "inbound_nodes": [[["Dense_layer_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_layer_2", "trainable": true, "dtype": "float32", "units": 150, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_layer_2", "inbound_nodes": [[["dropout_122", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_123", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_123", "inbound_nodes": [[["Dense_layer_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_layer_3", "trainable": true, "dtype": "float32", "units": 100, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_layer_3", "inbound_nodes": [[["dropout_123", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_124", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_124", "inbound_nodes": [[["Dense_layer_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_layer_4", "trainable": true, "dtype": "float32", "units": 100, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_layer_4", "inbound_nodes": [[["dropout_124", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_125", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_125", "inbound_nodes": [[["Dense_layer_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Logit_Probs", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Logit_Probs", "inbound_nodes": [[["dropout_125", 0, 0, {}]]]}], "input_layers": [["Input", 0, 0]], "output_layers": [["Logit_Probs", 0, 0]]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}}, "metrics": ["accuracy", {"class_name": "SparseTopKCategoricalAccuracy", "config": {"name": "sparse_top_k_categorical_accuracy", "dtype": "float32", "k": 3}}], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
é"æ
_tf_keras_input_layerÆ{"class_name": "InputLayer", "name": "Input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}}
È
regularization_losses
trainable_variables
	variables
	keras_api
+¨&call_and_return_all_conditional_losses
©__call__"·
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_121", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_121", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
¤

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+ª&call_and_return_all_conditional_losses
«__call__"ý
_tf_keras_layerã{"class_name": "Dense", "name": "Dense_layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Dense_layer_1", "trainable": true, "dtype": "float32", "units": 200, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0010000000474974513, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
È
regularization_losses
trainable_variables
	variables
	keras_api
+¬&call_and_return_all_conditional_losses
­__call__"·
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_122", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_122", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}


 kernel
!bias
"regularization_losses
#trainable_variables
$	variables
%	keras_api
+®&call_and_return_all_conditional_losses
¯__call__"ì
_tf_keras_layerÒ{"class_name": "Dense", "name": "Dense_layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Dense_layer_2", "trainable": true, "dtype": "float32", "units": 150, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
È
&regularization_losses
'trainable_variables
(	variables
)	keras_api
+°&call_and_return_all_conditional_losses
±__call__"·
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_123", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_123", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}


*kernel
+bias
,regularization_losses
-trainable_variables
.	variables
/	keras_api
+²&call_and_return_all_conditional_losses
³__call__"ë
_tf_keras_layerÑ{"class_name": "Dense", "name": "Dense_layer_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Dense_layer_3", "trainable": true, "dtype": "float32", "units": 100, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 150}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 150]}}
È
0regularization_losses
1trainable_variables
2	variables
3	keras_api
+´&call_and_return_all_conditional_losses
µ__call__"·
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_124", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_124", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}


4kernel
5bias
6regularization_losses
7trainable_variables
8	variables
9	keras_api
+¶&call_and_return_all_conditional_losses
·__call__"ì
_tf_keras_layerÒ{"class_name": "Dense", "name": "Dense_layer_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Dense_layer_4", "trainable": true, "dtype": "float32", "units": 100, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
È
:regularization_losses
;trainable_variables
<	variables
=	keras_api
+¸&call_and_return_all_conditional_losses
¹__call__"·
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_125", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_125", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
Û

>kernel
?bias
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
+º&call_and_return_all_conditional_losses
»__call__"´
_tf_keras_layer{"class_name": "Dense", "name": "Logit_Probs", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Logit_Probs", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}

Diter

Ebeta_1

Fbeta_2
	Gdecay
Hlearning_ratemm m!m*m+m4m5m>m?mvv v!v*v+v 4v¡5v¢>v£?v¤"
	optimizer
@
¼0
½1
¾2
¿3"
trackable_list_wrapper
f
0
1
 2
!3
*4
+5
46
57
>8
?9"
trackable_list_wrapper
f
0
1
 2
!3
*4
+5
46
57
>8
?9"
trackable_list_wrapper
Î
regularization_losses

Ilayers
trainable_variables
	variables
Jlayer_regularization_losses
Klayer_metrics
Lmetrics
Mnon_trainable_variables
§__call__
¥_default_save_signature
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
-
Àserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
regularization_losses

Nlayers
Olayer_regularization_losses
trainable_variables
	variables
Player_metrics
Qmetrics
Rnon_trainable_variables
©__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
+:)
È2Dense_layer_1_55/kernel
$:"È2Dense_layer_1_55/bias
(
¼0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
regularization_losses

Slayers
Tlayer_regularization_losses
trainable_variables
	variables
Ulayer_metrics
Vmetrics
Wnon_trainable_variables
«__call__
+ª&call_and_return_all_conditional_losses
'ª"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
regularization_losses

Xlayers
Ylayer_regularization_losses
trainable_variables
	variables
Zlayer_metrics
[metrics
\non_trainable_variables
­__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
+:)
È2Dense_layer_2_44/kernel
$:"2Dense_layer_2_44/bias
(
½0"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
°
"regularization_losses

]layers
^layer_regularization_losses
#trainable_variables
$	variables
_layer_metrics
`metrics
anon_trainable_variables
¯__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
&regularization_losses

blayers
clayer_regularization_losses
'trainable_variables
(	variables
dlayer_metrics
emetrics
fnon_trainable_variables
±__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
*:(	d2Dense_layer_3_22/kernel
#:!d2Dense_layer_3_22/bias
(
¾0"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
°
,regularization_losses

glayers
hlayer_regularization_losses
-trainable_variables
.	variables
ilayer_metrics
jmetrics
knon_trainable_variables
³__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
0regularization_losses

llayers
mlayer_regularization_losses
1trainable_variables
2	variables
nlayer_metrics
ometrics
pnon_trainable_variables
µ__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses"
_generic_user_object
&:$dd2Dense_layer_4/kernel
 :d2Dense_layer_4/bias
(
¿0"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
°
6regularization_losses

qlayers
rlayer_regularization_losses
7trainable_variables
8	variables
slayer_metrics
tmetrics
unon_trainable_variables
·__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
:regularization_losses

vlayers
wlayer_regularization_losses
;trainable_variables
<	variables
xlayer_metrics
ymetrics
znon_trainable_variables
¹__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses"
_generic_user_object
':%d
2Logit_Probs_55/kernel
!:
2Logit_Probs_55/bias
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
°
@regularization_losses

{layers
|layer_regularization_losses
Atrainable_variables
B	variables
}layer_metrics
~metrics
non_trainable_variables
»__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
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
trackable_list_wrapper
 "
trackable_dict_wrapper
8
0
1
2"
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
¼0"
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
½0"
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
¾0"
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
¿0"
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
¿

total

count
	variables
	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}


total

count

_fn_kwargs
	variables
	keras_api"¿
_tf_keras_metric¤{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
¬

total

count

_fn_kwargs
	variables
	keras_api"à
_tf_keras_metricÅ{"class_name": "SparseTopKCategoricalAccuracy", "name": "sparse_top_k_categorical_accuracy", "dtype": "float32", "config": {"name": "sparse_top_k_categorical_accuracy", "dtype": "float32", "k": 3}}
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
0:.
È2Adam/Dense_layer_1_55/kernel/m
):'È2Adam/Dense_layer_1_55/bias/m
0:.
È2Adam/Dense_layer_2_44/kernel/m
):'2Adam/Dense_layer_2_44/bias/m
/:-	d2Adam/Dense_layer_3_22/kernel/m
(:&d2Adam/Dense_layer_3_22/bias/m
+:)dd2Adam/Dense_layer_4/kernel/m
%:#d2Adam/Dense_layer_4/bias/m
,:*d
2Adam/Logit_Probs_55/kernel/m
&:$
2Adam/Logit_Probs_55/bias/m
0:.
È2Adam/Dense_layer_1_55/kernel/v
):'È2Adam/Dense_layer_1_55/bias/v
0:.
È2Adam/Dense_layer_2_44/kernel/v
):'2Adam/Dense_layer_2_44/bias/v
/:-	d2Adam/Dense_layer_3_22/kernel/v
(:&d2Adam/Dense_layer_3_22/bias/v
+:)dd2Adam/Dense_layer_4/kernel/v
%:#d2Adam/Dense_layer_4/bias/v
,:*d
2Adam/Logit_Probs_55/kernel/v
&:$
2Adam/Logit_Probs_55/bias/v
à2Ý
#__inference__wrapped_model_14497574µ
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
annotationsª *%¢"
 
Inputÿÿÿÿÿÿÿÿÿ
ê2ç
G__inference_initModel_layer_call_and_return_conditional_losses_14498226
G__inference_initModel_layer_call_and_return_conditional_losses_14497895
G__inference_initModel_layer_call_and_return_conditional_losses_14497940
G__inference_initModel_layer_call_and_return_conditional_losses_14498290À
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
þ2û
,__inference_initModel_layer_call_fn_14498081
,__inference_initModel_layer_call_fn_14498315
,__inference_initModel_layer_call_fn_14498011
,__inference_initModel_layer_call_fn_14498340À
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
Ð2Í
I__inference_dropout_121_layer_call_and_return_conditional_losses_14498352
I__inference_dropout_121_layer_call_and_return_conditional_losses_14498357´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
.__inference_dropout_121_layer_call_fn_14498367
.__inference_dropout_121_layer_call_fn_14498362´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
õ2ò
K__inference_Dense_layer_1_layer_call_and_return_conditional_losses_14498394¢
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
Ú2×
0__inference_Dense_layer_1_layer_call_fn_14498403¢
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
Ð2Í
I__inference_dropout_122_layer_call_and_return_conditional_losses_14498420
I__inference_dropout_122_layer_call_and_return_conditional_losses_14498415´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
.__inference_dropout_122_layer_call_fn_14498425
.__inference_dropout_122_layer_call_fn_14498430´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
õ2ò
K__inference_Dense_layer_2_layer_call_and_return_conditional_losses_14498448¢
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
Ú2×
0__inference_Dense_layer_2_layer_call_fn_14498457¢
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
Ð2Í
I__inference_dropout_123_layer_call_and_return_conditional_losses_14498469
I__inference_dropout_123_layer_call_and_return_conditional_losses_14498474´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
.__inference_dropout_123_layer_call_fn_14498479
.__inference_dropout_123_layer_call_fn_14498484´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
õ2ò
K__inference_Dense_layer_3_layer_call_and_return_conditional_losses_14498497¢
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
Ú2×
0__inference_Dense_layer_3_layer_call_fn_14498506¢
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
Ð2Í
I__inference_dropout_124_layer_call_and_return_conditional_losses_14498523
I__inference_dropout_124_layer_call_and_return_conditional_losses_14498518´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
.__inference_dropout_124_layer_call_fn_14498533
.__inference_dropout_124_layer_call_fn_14498528´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
õ2ò
K__inference_Dense_layer_4_layer_call_and_return_conditional_losses_14498551¢
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
Ú2×
0__inference_Dense_layer_4_layer_call_fn_14498560¢
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
Ð2Í
I__inference_dropout_125_layer_call_and_return_conditional_losses_14498577
I__inference_dropout_125_layer_call_and_return_conditional_losses_14498572´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
.__inference_dropout_125_layer_call_fn_14498582
.__inference_dropout_125_layer_call_fn_14498587´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ó2ð
I__inference_Logit_Probs_layer_call_and_return_conditional_losses_14498597¢
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
Ø2Õ
.__inference_Logit_Probs_layer_call_fn_14498606¢
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
µ2²
__inference_loss_fn_0_14498619
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_1_14498624
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_2_14498629
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_3_14498634
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
3B1
&__inference_signature_wrapper_14498127Input­
K__inference_Dense_layer_1_layer_call_and_return_conditional_losses_14498394^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÈ
 
0__inference_Dense_layer_1_layer_call_fn_14498403Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÈ­
K__inference_Dense_layer_2_layer_call_and_return_conditional_losses_14498448^ !0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÈ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_Dense_layer_2_layer_call_fn_14498457Q !0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÈ
ª "ÿÿÿÿÿÿÿÿÿ¬
K__inference_Dense_layer_3_layer_call_and_return_conditional_losses_14498497]*+0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 
0__inference_Dense_layer_3_layer_call_fn_14498506P*+0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿd«
K__inference_Dense_layer_4_layer_call_and_return_conditional_losses_14498551\45/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 
0__inference_Dense_layer_4_layer_call_fn_14498560O45/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "ÿÿÿÿÿÿÿÿÿd©
I__inference_Logit_Probs_layer_call_and_return_conditional_losses_14498597\>?/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
.__inference_Logit_Probs_layer_call_fn_14498606O>?/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "ÿÿÿÿÿÿÿÿÿ

#__inference__wrapped_model_14497574x
 !*+45>?/¢,
%¢"
 
Inputÿÿÿÿÿÿÿÿÿ
ª "9ª6
4
Logit_Probs%"
Logit_Probsÿÿÿÿÿÿÿÿÿ
«
I__inference_dropout_121_layer_call_and_return_conditional_losses_14498352^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 «
I__inference_dropout_121_layer_call_and_return_conditional_losses_14498357^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_dropout_121_layer_call_fn_14498362Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_dropout_121_layer_call_fn_14498367Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ«
I__inference_dropout_122_layer_call_and_return_conditional_losses_14498415^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÈ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÈ
 «
I__inference_dropout_122_layer_call_and_return_conditional_losses_14498420^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÈ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÈ
 
.__inference_dropout_122_layer_call_fn_14498425Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÈ
p
ª "ÿÿÿÿÿÿÿÿÿÈ
.__inference_dropout_122_layer_call_fn_14498430Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÈ
p 
ª "ÿÿÿÿÿÿÿÿÿÈ«
I__inference_dropout_123_layer_call_and_return_conditional_losses_14498469^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 «
I__inference_dropout_123_layer_call_and_return_conditional_losses_14498474^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_dropout_123_layer_call_fn_14498479Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_dropout_123_layer_call_fn_14498484Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ©
I__inference_dropout_124_layer_call_and_return_conditional_losses_14498518\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿd
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 ©
I__inference_dropout_124_layer_call_and_return_conditional_losses_14498523\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿd
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 
.__inference_dropout_124_layer_call_fn_14498528O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿd
p
ª "ÿÿÿÿÿÿÿÿÿd
.__inference_dropout_124_layer_call_fn_14498533O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿd
p 
ª "ÿÿÿÿÿÿÿÿÿd©
I__inference_dropout_125_layer_call_and_return_conditional_losses_14498572\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿd
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 ©
I__inference_dropout_125_layer_call_and_return_conditional_losses_14498577\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿd
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 
.__inference_dropout_125_layer_call_fn_14498582O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿd
p
ª "ÿÿÿÿÿÿÿÿÿd
.__inference_dropout_125_layer_call_fn_14498587O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿd
p 
ª "ÿÿÿÿÿÿÿÿÿd·
G__inference_initModel_layer_call_and_return_conditional_losses_14497895l
 !*+45>?7¢4
-¢*
 
Inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ·
G__inference_initModel_layer_call_and_return_conditional_losses_14497940l
 !*+45>?7¢4
-¢*
 
Inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ¸
G__inference_initModel_layer_call_and_return_conditional_losses_14498226m
 !*+45>?8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ¸
G__inference_initModel_layer_call_and_return_conditional_losses_14498290m
 !*+45>?8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
,__inference_initModel_layer_call_fn_14498011_
 !*+45>?7¢4
-¢*
 
Inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ

,__inference_initModel_layer_call_fn_14498081_
 !*+45>?7¢4
-¢*
 
Inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ

,__inference_initModel_layer_call_fn_14498315`
 !*+45>?8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ

,__inference_initModel_layer_call_fn_14498340`
 !*+45>?8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
=
__inference_loss_fn_0_14498619¢

¢ 
ª " :
__inference_loss_fn_1_14498624¢

¢ 
ª " :
__inference_loss_fn_2_14498629¢

¢ 
ª " :
__inference_loss_fn_3_14498634¢

¢ 
ª " ¬
&__inference_signature_wrapper_14498127
 !*+45>?8¢5
¢ 
.ª+
)
Input 
Inputÿÿÿÿÿÿÿÿÿ"9ª6
4
Logit_Probs%"
Logit_Probsÿÿÿÿÿÿÿÿÿ
