??
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
shapeshape?"serve*2.2.0-dlenv2v2.2.0-0-g2b96f368??
?
conv2d_54/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_54/kernel
}
$conv2d_54/kernel/Read/ReadVariableOpReadVariableOpconv2d_54/kernel*&
_output_shapes
: *
dtype0
t
conv2d_54/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_54/bias
m
"conv2d_54/bias/Read/ReadVariableOpReadVariableOpconv2d_54/bias*
_output_shapes
: *
dtype0
?
batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_10/gamma
?
0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_10/beta
?
/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_10/moving_mean
?
6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_10/moving_variance
?
:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_55/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_55/kernel
}
$conv2d_55/kernel/Read/ReadVariableOpReadVariableOpconv2d_55/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_55/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_55/bias
m
"conv2d_55/bias/Read/ReadVariableOpReadVariableOpconv2d_55/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_11/gamma
?
0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_11/beta
?
/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes
:@*
dtype0
?
"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_11/moving_mean
?
6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes
:@*
dtype0
?
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_11/moving_variance
?
:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes
:@*
dtype0
?
Dense_Layer_1_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*(
shared_nameDense_Layer_1_38/kernel
?
+Dense_Layer_1_38/kernel/Read/ReadVariableOpReadVariableOpDense_Layer_1_38/kernel*
_output_shapes
:	?d*
dtype0
?
Dense_Layer_1_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameDense_Layer_1_38/bias
{
)Dense_Layer_1_38/bias/Read/ReadVariableOpReadVariableOpDense_Layer_1_38/bias*
_output_shapes
:d*
dtype0
?
Logit_Probs_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*&
shared_nameLogit_Probs_38/kernel

)Logit_Probs_38/kernel/Read/ReadVariableOpReadVariableOpLogit_Probs_38/kernel*
_output_shapes

:d
*
dtype0
~
Logit_Probs_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameLogit_Probs_38/bias
w
'Logit_Probs_38/bias/Read/ReadVariableOpReadVariableOpLogit_Probs_38/bias*
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
Adam/conv2d_54/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_54/kernel/m
?
+Adam/conv2d_54/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_54/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_54/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_54/bias/m
{
)Adam/conv2d_54/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_54/bias/m*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_10/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_10/gamma/m
?
7Adam/batch_normalization_10/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_10/gamma/m*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_10/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_10/beta/m
?
6Adam/batch_normalization_10/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_10/beta/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_55/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_55/kernel/m
?
+Adam/conv2d_55/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_55/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_55/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_55/bias/m
{
)Adam/conv2d_55/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_55/bias/m*
_output_shapes
:@*
dtype0
?
#Adam/batch_normalization_11/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_11/gamma/m
?
7Adam/batch_normalization_11/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_11/gamma/m*
_output_shapes
:@*
dtype0
?
"Adam/batch_normalization_11/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_11/beta/m
?
6Adam/batch_normalization_11/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_11/beta/m*
_output_shapes
:@*
dtype0
?
Adam/Dense_Layer_1_38/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*/
shared_name Adam/Dense_Layer_1_38/kernel/m
?
2Adam/Dense_Layer_1_38/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_Layer_1_38/kernel/m*
_output_shapes
:	?d*
dtype0
?
Adam/Dense_Layer_1_38/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*-
shared_nameAdam/Dense_Layer_1_38/bias/m
?
0Adam/Dense_Layer_1_38/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_Layer_1_38/bias/m*
_output_shapes
:d*
dtype0
?
Adam/Logit_Probs_38/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*-
shared_nameAdam/Logit_Probs_38/kernel/m
?
0Adam/Logit_Probs_38/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Logit_Probs_38/kernel/m*
_output_shapes

:d
*
dtype0
?
Adam/Logit_Probs_38/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_nameAdam/Logit_Probs_38/bias/m
?
.Adam/Logit_Probs_38/bias/m/Read/ReadVariableOpReadVariableOpAdam/Logit_Probs_38/bias/m*
_output_shapes
:
*
dtype0
?
Adam/conv2d_54/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_54/kernel/v
?
+Adam/conv2d_54/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_54/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_54/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_54/bias/v
{
)Adam/conv2d_54/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_54/bias/v*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_10/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_10/gamma/v
?
7Adam/batch_normalization_10/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_10/gamma/v*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_10/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_10/beta/v
?
6Adam/batch_normalization_10/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_10/beta/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_55/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_55/kernel/v
?
+Adam/conv2d_55/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_55/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_55/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_55/bias/v
{
)Adam/conv2d_55/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_55/bias/v*
_output_shapes
:@*
dtype0
?
#Adam/batch_normalization_11/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_11/gamma/v
?
7Adam/batch_normalization_11/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_11/gamma/v*
_output_shapes
:@*
dtype0
?
"Adam/batch_normalization_11/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_11/beta/v
?
6Adam/batch_normalization_11/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_11/beta/v*
_output_shapes
:@*
dtype0
?
Adam/Dense_Layer_1_38/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*/
shared_name Adam/Dense_Layer_1_38/kernel/v
?
2Adam/Dense_Layer_1_38/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_Layer_1_38/kernel/v*
_output_shapes
:	?d*
dtype0
?
Adam/Dense_Layer_1_38/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*-
shared_nameAdam/Dense_Layer_1_38/bias/v
?
0Adam/Dense_Layer_1_38/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_Layer_1_38/bias/v*
_output_shapes
:d*
dtype0
?
Adam/Logit_Probs_38/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*-
shared_nameAdam/Logit_Probs_38/kernel/v
?
0Adam/Logit_Probs_38/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Logit_Probs_38/kernel/v*
_output_shapes

:d
*
dtype0
?
Adam/Logit_Probs_38/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_nameAdam/Logit_Probs_38/bias/v
?
.Adam/Logit_Probs_38/bias/v/Read/ReadVariableOpReadVariableOpAdam/Logit_Probs_38/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
?U
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?U
value?UB?U B?T
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
?
axis
	gamma
beta
 moving_mean
!moving_variance
"trainable_variables
#	variables
$regularization_losses
%	keras_api
R
&trainable_variables
'	variables
(regularization_losses
)	keras_api
h

*kernel
+bias
,trainable_variables
-	variables
.regularization_losses
/	keras_api
?
0axis
	1gamma
2beta
3moving_mean
4moving_variance
5trainable_variables
6	variables
7regularization_losses
8	keras_api
R
9trainable_variables
:	variables
;regularization_losses
<	keras_api
R
=trainable_variables
>	variables
?regularization_losses
@	keras_api
R
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
h

Ekernel
Fbias
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
h

Kkernel
Lbias
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
?
Qiter

Rbeta_1

Sbeta_2
	Tdecay
Ulearning_ratem?m?m?m?*m?+m?1m?2m?Em?Fm?Km?Lm?v?v?v?v?*v?+v?1v?2v?Ev?Fv?Kv?Lv?
V
0
1
2
3
*4
+5
16
27
E8
F9
K10
L11
v
0
1
2
3
 4
!5
*6
+7
18
29
310
411
E12
F13
K14
L15
 
?
Vmetrics
Wlayer_regularization_losses
trainable_variables

Xlayers
Ynon_trainable_variables
	variables
Zlayer_metrics
regularization_losses
 
 
 
 
?
[metrics
\layer_regularization_losses

]layers
^non_trainable_variables
trainable_variables
	variables
_layer_metrics
regularization_losses
\Z
VARIABLE_VALUEconv2d_54/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_54/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
`metrics
alayer_regularization_losses

blayers
cnon_trainable_variables
trainable_variables
	variables
dlayer_metrics
regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_10/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_10/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_10/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_10/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 2
!3
 
?
emetrics
flayer_regularization_losses

glayers
hnon_trainable_variables
"trainable_variables
#	variables
ilayer_metrics
$regularization_losses
 
 
 
?
jmetrics
klayer_regularization_losses

llayers
mnon_trainable_variables
&trainable_variables
'	variables
nlayer_metrics
(regularization_losses
\Z
VARIABLE_VALUEconv2d_55/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_55/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1

*0
+1
 
?
ometrics
player_regularization_losses

qlayers
rnon_trainable_variables
,trainable_variables
-	variables
slayer_metrics
.regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_11/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_11/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_11/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_11/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

10
21

10
21
32
43
 
?
tmetrics
ulayer_regularization_losses

vlayers
wnon_trainable_variables
5trainable_variables
6	variables
xlayer_metrics
7regularization_losses
 
 
 
?
ymetrics
zlayer_regularization_losses

{layers
|non_trainable_variables
9trainable_variables
:	variables
}layer_metrics
;regularization_losses
 
 
 
?
~metrics
layer_regularization_losses
?layers
?non_trainable_variables
=trainable_variables
>	variables
?layer_metrics
?regularization_losses
 
 
 
?
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
Atrainable_variables
B	variables
?layer_metrics
Cregularization_losses
ca
VARIABLE_VALUEDense_Layer_1_38/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEDense_Layer_1_38/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

E0
F1

E0
F1
 
?
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
Gtrainable_variables
H	variables
?layer_metrics
Iregularization_losses
a_
VARIABLE_VALUELogit_Probs_38/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUELogit_Probs_38/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

K0
L1

K0
L1
 
?
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
Mtrainable_variables
N	variables
?layer_metrics
Oregularization_losses
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

?0
?1
?2
 
V
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
11

 0
!1
32
43
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

 0
!1
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

30
41
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

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
}
VARIABLE_VALUEAdam/conv2d_54/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_54/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_10/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_10/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_55/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_55/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_11/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_11/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Dense_Layer_1_38/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Dense_Layer_1_38/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Logit_Probs_38/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/Logit_Probs_38/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_54/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_54/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_10/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_10/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_55/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_55/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_11/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_11/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Dense_Layer_1_38/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Dense_Layer_1_38/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Logit_Probs_38/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/Logit_Probs_38/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_InputPlaceholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_Inputconv2d_54/kernelconv2d_54/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_55/kernelconv2d_55/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceDense_Layer_1_38/kernelDense_Layer_1_38/biasLogit_Probs_38/kernelLogit_Probs_38/bias*
Tin
2*
Tout
2*'
_output_shapes
:?????????
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*.
f)R'
%__inference_signature_wrapper_5194196
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_54/kernel/Read/ReadVariableOp"conv2d_54/bias/Read/ReadVariableOp0batch_normalization_10/gamma/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp$conv2d_55/kernel/Read/ReadVariableOp"conv2d_55/bias/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOp+Dense_Layer_1_38/kernel/Read/ReadVariableOp)Dense_Layer_1_38/bias/Read/ReadVariableOp)Logit_Probs_38/kernel/Read/ReadVariableOp'Logit_Probs_38/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp+Adam/conv2d_54/kernel/m/Read/ReadVariableOp)Adam/conv2d_54/bias/m/Read/ReadVariableOp7Adam/batch_normalization_10/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_10/beta/m/Read/ReadVariableOp+Adam/conv2d_55/kernel/m/Read/ReadVariableOp)Adam/conv2d_55/bias/m/Read/ReadVariableOp7Adam/batch_normalization_11/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_11/beta/m/Read/ReadVariableOp2Adam/Dense_Layer_1_38/kernel/m/Read/ReadVariableOp0Adam/Dense_Layer_1_38/bias/m/Read/ReadVariableOp0Adam/Logit_Probs_38/kernel/m/Read/ReadVariableOp.Adam/Logit_Probs_38/bias/m/Read/ReadVariableOp+Adam/conv2d_54/kernel/v/Read/ReadVariableOp)Adam/conv2d_54/bias/v/Read/ReadVariableOp7Adam/batch_normalization_10/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_10/beta/v/Read/ReadVariableOp+Adam/conv2d_55/kernel/v/Read/ReadVariableOp)Adam/conv2d_55/bias/v/Read/ReadVariableOp7Adam/batch_normalization_11/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_11/beta/v/Read/ReadVariableOp2Adam/Dense_Layer_1_38/kernel/v/Read/ReadVariableOp0Adam/Dense_Layer_1_38/bias/v/Read/ReadVariableOp0Adam/Logit_Probs_38/kernel/v/Read/ReadVariableOp.Adam/Logit_Probs_38/bias/v/Read/ReadVariableOpConst*@
Tin9
725	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*)
f$R"
 __inference__traced_save_5195051
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_54/kernelconv2d_54/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_55/kernelconv2d_55/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceDense_Layer_1_38/kernelDense_Layer_1_38/biasLogit_Probs_38/kernelLogit_Probs_38/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/conv2d_54/kernel/mAdam/conv2d_54/bias/m#Adam/batch_normalization_10/gamma/m"Adam/batch_normalization_10/beta/mAdam/conv2d_55/kernel/mAdam/conv2d_55/bias/m#Adam/batch_normalization_11/gamma/m"Adam/batch_normalization_11/beta/mAdam/Dense_Layer_1_38/kernel/mAdam/Dense_Layer_1_38/bias/mAdam/Logit_Probs_38/kernel/mAdam/Logit_Probs_38/bias/mAdam/conv2d_54/kernel/vAdam/conv2d_54/bias/v#Adam/batch_normalization_10/gamma/v"Adam/batch_normalization_10/beta/vAdam/conv2d_55/kernel/vAdam/conv2d_55/bias/v#Adam/batch_normalization_11/gamma/v"Adam/batch_normalization_11/beta/vAdam/Dense_Layer_1_38/kernel/vAdam/Dense_Layer_1_38/bias/vAdam/Logit_Probs_38/kernel/vAdam/Logit_Probs_38/bias/v*?
Tin8
624*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*,
f'R%
#__inference__traced_restore_5195216??
?8
?
B__inference_Conv4_layer_call_and_return_conditional_losses_5194006

inputs
conv2d_54_5193963
conv2d_54_5193965"
batch_normalization_10_5193968"
batch_normalization_10_5193970"
batch_normalization_10_5193972"
batch_normalization_10_5193974
conv2d_55_5193978
conv2d_55_5193980"
batch_normalization_11_5193983"
batch_normalization_11_5193985"
batch_normalization_11_5193987"
batch_normalization_11_5193989
dense_layer_1_5193995
dense_layer_1_5193997
logit_probs_5194000
logit_probs_5194002
identity??%Dense_Layer_1/StatefulPartitionedCall?#Logit_Probs/StatefulPartitionedCall?.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?!conv2d_54/StatefulPartitionedCall?!conv2d_55/StatefulPartitionedCall?"dropout_16/StatefulPartitionedCall?
reshape_38/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_reshape_38_layer_call_and_return_conditional_losses_51936232
reshape_38/PartitionedCall?
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCall#reshape_38/PartitionedCall:output:0conv2d_54_5193963conv2d_54_5193965*
Tin
2*
Tout
2*/
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_54_layer_call_and_return_conditional_losses_51932972#
!conv2d_54/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0batch_normalization_10_5193968batch_normalization_10_5193970batch_normalization_10_5193972batch_normalization_10_5193974*
Tin	
2*
Tout
2*/
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_519366620
.batch_normalization_10/StatefulPartitionedCall?
 max_pooling2d_54/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_51934392"
 max_pooling2d_54/PartitionedCall?
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_54/PartitionedCall:output:0conv2d_55_5193978conv2d_55_5193980*
Tin
2*
Tout
2*/
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_55_layer_call_and_return_conditional_losses_51934572#
!conv2d_55/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0batch_normalization_11_5193983batch_normalization_11_5193985batch_normalization_11_5193987batch_normalization_11_5193989*
Tin	
2*
Tout
2*/
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_519375620
.batch_normalization_11/StatefulPartitionedCall?
 max_pooling2d_55/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_51935992"
 max_pooling2d_55/PartitionedCall?
flatten_38/PartitionedCallPartitionedCall)max_pooling2d_55/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_flatten_38_layer_call_and_return_conditional_losses_51938172
flatten_38/PartitionedCall?
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall#flatten_38/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dropout_16_layer_call_and_return_conditional_losses_51938372$
"dropout_16/StatefulPartitionedCall?
%Dense_Layer_1/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0dense_layer_1_5193995dense_layer_1_5193997*
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
CPU

GPU2*0J 8*S
fNRL
J__inference_Dense_Layer_1_layer_call_and_return_conditional_losses_51938662'
%Dense_Layer_1/StatefulPartitionedCall?
#Logit_Probs/StatefulPartitionedCallStatefulPartitionedCall.Dense_Layer_1/StatefulPartitionedCall:output:0logit_probs_5194000logit_probs_5194002*
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
CPU

GPU2*0J 8*Q
fLRJ
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_51938922%
#Logit_Probs/StatefulPartitionedCall?
IdentityIdentity,Logit_Probs/StatefulPartitionedCall:output:0&^Dense_Layer_1/StatefulPartitionedCall$^Logit_Probs/StatefulPartitionedCall/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:??????????::::::::::::::::2N
%Dense_Layer_1/StatefulPartitionedCall%Dense_Layer_1/StatefulPartitionedCall2J
#Logit_Probs/StatefulPartitionedCall#Logit_Probs/StatefulPartitionedCall2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall:P L
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
: 
?
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_5194693

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity?t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@:::::i e
A
_output_shapes/
-:+???????????????????????????@
 
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
: 
?
f
G__inference_dropout_16_layer_call_and_return_conditional_losses_5194817

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
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
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
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?P
?
B__inference_Conv4_layer_call_and_return_conditional_losses_5194377

inputs,
(conv2d_54_conv2d_readvariableop_resource-
)conv2d_54_biasadd_readvariableop_resource2
.batch_normalization_10_readvariableop_resource4
0batch_normalization_10_readvariableop_1_resourceC
?batch_normalization_10_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_55_conv2d_readvariableop_resource-
)conv2d_55_biasadd_readvariableop_resource2
.batch_normalization_11_readvariableop_resource4
0batch_normalization_11_readvariableop_1_resourceC
?batch_normalization_11_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource0
,dense_layer_1_matmul_readvariableop_resource1
-dense_layer_1_biasadd_readvariableop_resource.
*logit_probs_matmul_readvariableop_resource/
+logit_probs_biasadd_readvariableop_resource
identity?Z
reshape_38/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_38/Shape?
reshape_38/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_38/strided_slice/stack?
 reshape_38/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_38/strided_slice/stack_1?
 reshape_38/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_38/strided_slice/stack_2?
reshape_38/strided_sliceStridedSlicereshape_38/Shape:output:0'reshape_38/strided_slice/stack:output:0)reshape_38/strided_slice/stack_1:output:0)reshape_38/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_38/strided_slicez
reshape_38/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_38/Reshape/shape/1z
reshape_38/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_38/Reshape/shape/2z
reshape_38/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_38/Reshape/shape/3?
reshape_38/Reshape/shapePack!reshape_38/strided_slice:output:0#reshape_38/Reshape/shape/1:output:0#reshape_38/Reshape/shape/2:output:0#reshape_38/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_38/Reshape/shape?
reshape_38/ReshapeReshapeinputs!reshape_38/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape_38/Reshape?
conv2d_54/Conv2D/ReadVariableOpReadVariableOp(conv2d_54_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_54/Conv2D/ReadVariableOp?
conv2d_54/Conv2DConv2Dreshape_38/Reshape:output:0'conv2d_54/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d_54/Conv2D?
 conv2d_54/BiasAdd/ReadVariableOpReadVariableOp)conv2d_54_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_54/BiasAdd/ReadVariableOp?
conv2d_54/BiasAddBiasAddconv2d_54/Conv2D:output:0(conv2d_54/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_54/BiasAdd~
conv2d_54/ReluReluconv2d_54/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_54/Relu?
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_10/ReadVariableOp?
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_10/ReadVariableOp_1?
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_54/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( 2)
'batch_normalization_10/FusedBatchNormV3?
max_pooling2d_54/MaxPoolMaxPool+batch_normalization_10/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_54/MaxPool?
conv2d_55/Conv2D/ReadVariableOpReadVariableOp(conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_55/Conv2D/ReadVariableOp?
conv2d_55/Conv2DConv2D!max_pooling2d_54/MaxPool:output:0'conv2d_55/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_55/Conv2D?
 conv2d_55/BiasAdd/ReadVariableOpReadVariableOp)conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_55/BiasAdd/ReadVariableOp?
conv2d_55/BiasAddBiasAddconv2d_55/Conv2D:output:0(conv2d_55/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_55/BiasAdd~
conv2d_55/ReluReluconv2d_55/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_55/Relu?
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_11/ReadVariableOp?
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_11/ReadVariableOp_1?
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_55/Relu:activations:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2)
'batch_normalization_11/FusedBatchNormV3?
max_pooling2d_55/MaxPoolMaxPool+batch_normalization_11/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_55/MaxPoolu
flatten_38/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
flatten_38/Const?
flatten_38/ReshapeReshape!max_pooling2d_55/MaxPool:output:0flatten_38/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_38/Reshape?
dropout_16/IdentityIdentityflatten_38/Reshape:output:0*
T0*(
_output_shapes
:??????????2
dropout_16/Identity?
#Dense_Layer_1/MatMul/ReadVariableOpReadVariableOp,dense_layer_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02%
#Dense_Layer_1/MatMul/ReadVariableOp?
Dense_Layer_1/MatMulMatMuldropout_16/Identity:output:0+Dense_Layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
Dense_Layer_1/MatMul?
$Dense_Layer_1/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02&
$Dense_Layer_1/BiasAdd/ReadVariableOp?
Dense_Layer_1/BiasAddBiasAddDense_Layer_1/MatMul:product:0,Dense_Layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
Dense_Layer_1/BiasAdd?
Dense_Layer_1/ReluReluDense_Layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Dense_Layer_1/Relu?
!Logit_Probs/MatMul/ReadVariableOpReadVariableOp*logit_probs_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02#
!Logit_Probs/MatMul/ReadVariableOp?
Logit_Probs/MatMulMatMul Dense_Layer_1/Relu:activations:0)Logit_Probs/MatMul/ReadVariableOp:value:0*
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
identityIdentity:output:0*g
_input_shapesV
T:??????????:::::::::::::::::P L
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
: 
??
?
#__inference__traced_restore_5195216
file_prefix%
!assignvariableop_conv2d_54_kernel%
!assignvariableop_1_conv2d_54_bias3
/assignvariableop_2_batch_normalization_10_gamma2
.assignvariableop_3_batch_normalization_10_beta9
5assignvariableop_4_batch_normalization_10_moving_mean=
9assignvariableop_5_batch_normalization_10_moving_variance'
#assignvariableop_6_conv2d_55_kernel%
!assignvariableop_7_conv2d_55_bias3
/assignvariableop_8_batch_normalization_11_gamma2
.assignvariableop_9_batch_normalization_11_beta:
6assignvariableop_10_batch_normalization_11_moving_mean>
:assignvariableop_11_batch_normalization_11_moving_variance/
+assignvariableop_12_dense_layer_1_38_kernel-
)assignvariableop_13_dense_layer_1_38_bias-
)assignvariableop_14_logit_probs_38_kernel+
'assignvariableop_15_logit_probs_38_bias!
assignvariableop_16_adam_iter#
assignvariableop_17_adam_beta_1#
assignvariableop_18_adam_beta_2"
assignvariableop_19_adam_decay*
&assignvariableop_20_adam_learning_rate
assignvariableop_21_total
assignvariableop_22_count
assignvariableop_23_total_1
assignvariableop_24_count_1
assignvariableop_25_total_2
assignvariableop_26_count_2/
+assignvariableop_27_adam_conv2d_54_kernel_m-
)assignvariableop_28_adam_conv2d_54_bias_m;
7assignvariableop_29_adam_batch_normalization_10_gamma_m:
6assignvariableop_30_adam_batch_normalization_10_beta_m/
+assignvariableop_31_adam_conv2d_55_kernel_m-
)assignvariableop_32_adam_conv2d_55_bias_m;
7assignvariableop_33_adam_batch_normalization_11_gamma_m:
6assignvariableop_34_adam_batch_normalization_11_beta_m6
2assignvariableop_35_adam_dense_layer_1_38_kernel_m4
0assignvariableop_36_adam_dense_layer_1_38_bias_m4
0assignvariableop_37_adam_logit_probs_38_kernel_m2
.assignvariableop_38_adam_logit_probs_38_bias_m/
+assignvariableop_39_adam_conv2d_54_kernel_v-
)assignvariableop_40_adam_conv2d_54_bias_v;
7assignvariableop_41_adam_batch_normalization_10_gamma_v:
6assignvariableop_42_adam_batch_normalization_10_beta_v/
+assignvariableop_43_adam_conv2d_55_kernel_v-
)assignvariableop_44_adam_conv2d_55_bias_v;
7assignvariableop_45_adam_batch_normalization_11_gamma_v:
6assignvariableop_46_adam_batch_normalization_11_beta_v6
2assignvariableop_47_adam_dense_layer_1_38_kernel_v4
0assignvariableop_48_adam_dense_layer_1_38_bias_v4
0assignvariableop_49_adam_logit_probs_38_kernel_v2
.assignvariableop_50_adam_logit_probs_38_bias_v
identity_52??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*?
value?B?3B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::*A
dtypes7
523	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_54_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_54_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_10_gammaIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_10_betaIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_10_moving_meanIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_10_moving_varianceIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_55_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_55_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_11_gammaIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_11_betaIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_11_moving_meanIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_11_moving_varianceIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp+assignvariableop_12_dense_layer_1_38_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp)assignvariableop_13_dense_layer_1_38_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp)assignvariableop_14_logit_probs_38_kernelIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp'assignvariableop_15_logit_probs_38_biasIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0	*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_2Identity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_2Identity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_54_kernel_mIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_54_bias_mIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp7assignvariableop_29_adam_batch_normalization_10_gamma_mIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp6assignvariableop_30_adam_batch_normalization_10_beta_mIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_55_kernel_mIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_55_bias_mIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp7assignvariableop_33_adam_batch_normalization_11_gamma_mIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp6assignvariableop_34_adam_batch_normalization_11_beta_mIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp2assignvariableop_35_adam_dense_layer_1_38_kernel_mIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp0assignvariableop_36_adam_dense_layer_1_38_bias_mIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp0assignvariableop_37_adam_logit_probs_38_kernel_mIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp.assignvariableop_38_adam_logit_probs_38_bias_mIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_54_kernel_vIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_54_bias_vIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp7assignvariableop_41_adam_batch_normalization_10_gamma_vIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp6assignvariableop_42_adam_batch_normalization_10_beta_vIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_55_kernel_vIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_55_bias_vIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp7assignvariableop_45_adam_batch_normalization_11_gamma_vIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp6assignvariableop_46_adam_batch_normalization_11_beta_vIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp2assignvariableop_47_adam_dense_layer_1_38_kernel_vIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp0assignvariableop_48_adam_dense_layer_1_38_bias_vIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp0assignvariableop_49_adam_logit_probs_38_kernel_vIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp.assignvariableop_50_adam_logit_probs_38_bias_vIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50?
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
NoOp?	
Identity_51Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_51?	
Identity_52IdentityIdentity_51:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_52"#
identity_52Identity_52:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502(
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
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: 
?
c
G__inference_flatten_38_layer_call_and_return_conditional_losses_5193817

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
F__inference_conv2d_55_layer_call_and_return_conditional_losses_5193457

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? :::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_5193774

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity?t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????@:::::W S
/
_output_shapes
:?????????@
 
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
: 
?
?
8__inference_batch_normalization_11_layer_call_fn_5194719

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_51935822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
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
: 
?
?
'__inference_Conv4_layer_call_fn_5194125	
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
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*'
_output_shapes
:?????????
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_Conv4_layer_call_and_return_conditional_losses_51940902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:??????????::::::::::::::::22
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
: 
?
c
G__inference_reshape_38_layer_call_and_return_conditional_losses_5193623

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
H
,__inference_reshape_38_layer_call_fn_5194470

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_reshape_38_layer_call_and_return_conditional_losses_51936232
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?$
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_5194750

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:?????????@
 
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
: 
?
?
-__inference_Logit_Probs_layer_call_fn_5194871

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
CPU

GPU2*0J 8*Q
fLRJ
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_51938922
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
?
?
'__inference_Conv4_layer_call_fn_5194041	
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
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_Conv4_layer_call_and_return_conditional_losses_51940062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:??????????::::::::::::::::22
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
: 
?
e
G__inference_dropout_16_layer_call_and_return_conditional_losses_5194822

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?$
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_5194588

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:????????? 
 
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
: 
?
?
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_5194862

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
8__inference_batch_normalization_11_layer_call_fn_5194794

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:?????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_51937742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
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
: 
?
?
/__inference_Dense_Layer_1_layer_call_fn_5194852

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
CPU

GPU2*0J 8*S
fNRL
J__inference_Dense_Layer_1_layer_call_and_return_conditional_losses_51938662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?$
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_5194675

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
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
: 
?
?
+__inference_conv2d_55_layer_call_fn_5193467

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_55_layer_call_and_return_conditional_losses_51934572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_5193684

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity?t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? :::::W S
/
_output_shapes
:????????? 
 
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
: 
??
?	
B__inference_Conv4_layer_call_and_return_conditional_losses_5194303

inputs,
(conv2d_54_conv2d_readvariableop_resource-
)conv2d_54_biasadd_readvariableop_resource2
.batch_normalization_10_readvariableop_resource4
0batch_normalization_10_readvariableop_1_resourceC
?batch_normalization_10_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_55_conv2d_readvariableop_resource-
)conv2d_55_biasadd_readvariableop_resource2
.batch_normalization_11_readvariableop_resource4
0batch_normalization_11_readvariableop_1_resourceC
?batch_normalization_11_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource0
,dense_layer_1_matmul_readvariableop_resource1
-dense_layer_1_biasadd_readvariableop_resource.
*logit_probs_matmul_readvariableop_resource/
+logit_probs_biasadd_readvariableop_resource
identity??:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp?<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp?:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp?<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOpZ
reshape_38/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_38/Shape?
reshape_38/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_38/strided_slice/stack?
 reshape_38/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_38/strided_slice/stack_1?
 reshape_38/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_38/strided_slice/stack_2?
reshape_38/strided_sliceStridedSlicereshape_38/Shape:output:0'reshape_38/strided_slice/stack:output:0)reshape_38/strided_slice/stack_1:output:0)reshape_38/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_38/strided_slicez
reshape_38/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_38/Reshape/shape/1z
reshape_38/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_38/Reshape/shape/2z
reshape_38/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_38/Reshape/shape/3?
reshape_38/Reshape/shapePack!reshape_38/strided_slice:output:0#reshape_38/Reshape/shape/1:output:0#reshape_38/Reshape/shape/2:output:0#reshape_38/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_38/Reshape/shape?
reshape_38/ReshapeReshapeinputs!reshape_38/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape_38/Reshape?
conv2d_54/Conv2D/ReadVariableOpReadVariableOp(conv2d_54_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_54/Conv2D/ReadVariableOp?
conv2d_54/Conv2DConv2Dreshape_38/Reshape:output:0'conv2d_54/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d_54/Conv2D?
 conv2d_54/BiasAdd/ReadVariableOpReadVariableOp)conv2d_54_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_54/BiasAdd/ReadVariableOp?
conv2d_54/BiasAddBiasAddconv2d_54/Conv2D:output:0(conv2d_54/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_54/BiasAdd~
conv2d_54/ReluReluconv2d_54/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_54/Relu?
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_10/ReadVariableOp?
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_10/ReadVariableOp_1?
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_54/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:2)
'batch_normalization_10/FusedBatchNormV3?
batch_normalization_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
batch_normalization_10/Const?
,batch_normalization_10/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,batch_normalization_10/AssignMovingAvg/sub/x?
*batch_normalization_10/AssignMovingAvg/subSub5batch_normalization_10/AssignMovingAvg/sub/x:output:0%batch_normalization_10/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_10/AssignMovingAvg/sub?
5batch_normalization_10/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_10/AssignMovingAvg/ReadVariableOp?
,batch_normalization_10/AssignMovingAvg/sub_1Sub=batch_normalization_10/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_10/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2.
,batch_normalization_10/AssignMovingAvg/sub_1?
*batch_normalization_10/AssignMovingAvg/mulMul0batch_normalization_10/AssignMovingAvg/sub_1:z:0.batch_normalization_10/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_10/AssignMovingAvg/mul?
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource.batch_normalization_10/AssignMovingAvg/mul:z:06^batch_normalization_10/AssignMovingAvg/ReadVariableOp7^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_10/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??20
.batch_normalization_10/AssignMovingAvg_1/sub/x?
,batch_normalization_10/AssignMovingAvg_1/subSub7batch_normalization_10/AssignMovingAvg_1/sub/x:output:0%batch_normalization_10/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_10/AssignMovingAvg_1/sub?
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp?
.batch_normalization_10/AssignMovingAvg_1/sub_1Sub?batch_normalization_10/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_10/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 20
.batch_normalization_10/AssignMovingAvg_1/sub_1?
,batch_normalization_10/AssignMovingAvg_1/mulMul2batch_normalization_10/AssignMovingAvg_1/sub_1:z:00batch_normalization_10/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_10/AssignMovingAvg_1/mul?
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_10/AssignMovingAvg_1/mul:z:08^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp?
max_pooling2d_54/MaxPoolMaxPool+batch_normalization_10/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_54/MaxPool?
conv2d_55/Conv2D/ReadVariableOpReadVariableOp(conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_55/Conv2D/ReadVariableOp?
conv2d_55/Conv2DConv2D!max_pooling2d_54/MaxPool:output:0'conv2d_55/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_55/Conv2D?
 conv2d_55/BiasAdd/ReadVariableOpReadVariableOp)conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_55/BiasAdd/ReadVariableOp?
conv2d_55/BiasAddBiasAddconv2d_55/Conv2D:output:0(conv2d_55/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_55/BiasAdd~
conv2d_55/ReluReluconv2d_55/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_55/Relu?
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_11/ReadVariableOp?
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_11/ReadVariableOp_1?
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_55/Relu:activations:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:2)
'batch_normalization_11/FusedBatchNormV3?
batch_normalization_11/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
batch_normalization_11/Const?
,batch_normalization_11/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,batch_normalization_11/AssignMovingAvg/sub/x?
*batch_normalization_11/AssignMovingAvg/subSub5batch_normalization_11/AssignMovingAvg/sub/x:output:0%batch_normalization_11/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_11/AssignMovingAvg/sub?
5batch_normalization_11/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_11/AssignMovingAvg/ReadVariableOp?
,batch_normalization_11/AssignMovingAvg/sub_1Sub=batch_normalization_11/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_11/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2.
,batch_normalization_11/AssignMovingAvg/sub_1?
*batch_normalization_11/AssignMovingAvg/mulMul0batch_normalization_11/AssignMovingAvg/sub_1:z:0.batch_normalization_11/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2,
*batch_normalization_11/AssignMovingAvg/mul?
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource.batch_normalization_11/AssignMovingAvg/mul:z:06^batch_normalization_11/AssignMovingAvg/ReadVariableOp7^batch_normalization_11/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_11/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??20
.batch_normalization_11/AssignMovingAvg_1/sub/x?
,batch_normalization_11/AssignMovingAvg_1/subSub7batch_normalization_11/AssignMovingAvg_1/sub/x:output:0%batch_normalization_11/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_11/AssignMovingAvg_1/sub?
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp?
.batch_normalization_11/AssignMovingAvg_1/sub_1Sub?batch_normalization_11/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_11/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@20
.batch_normalization_11/AssignMovingAvg_1/sub_1?
,batch_normalization_11/AssignMovingAvg_1/mulMul2batch_normalization_11/AssignMovingAvg_1/sub_1:z:00batch_normalization_11/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2.
,batch_normalization_11/AssignMovingAvg_1/mul?
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_11/AssignMovingAvg_1/mul:z:08^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp?
max_pooling2d_55/MaxPoolMaxPool+batch_normalization_11/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_55/MaxPoolu
flatten_38/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
flatten_38/Const?
flatten_38/ReshapeReshape!max_pooling2d_55/MaxPool:output:0flatten_38/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_38/Reshapey
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_16/dropout/Const?
dropout_16/dropout/MulMulflatten_38/Reshape:output:0!dropout_16/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_16/dropout/Mul
dropout_16/dropout/ShapeShapeflatten_38/Reshape:output:0*
T0*
_output_shapes
:2
dropout_16/dropout/Shape?
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*

seed21
/dropout_16/dropout/random_uniform/RandomUniform?
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_16/dropout/GreaterEqual/y?
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
dropout_16/dropout/GreaterEqual?
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_16/dropout/Cast?
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_16/dropout/Mul_1?
#Dense_Layer_1/MatMul/ReadVariableOpReadVariableOp,dense_layer_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02%
#Dense_Layer_1/MatMul/ReadVariableOp?
Dense_Layer_1/MatMulMatMuldropout_16/dropout/Mul_1:z:0+Dense_Layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
Dense_Layer_1/MatMul?
$Dense_Layer_1/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02&
$Dense_Layer_1/BiasAdd/ReadVariableOp?
Dense_Layer_1/BiasAddBiasAddDense_Layer_1/MatMul:product:0,Dense_Layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
Dense_Layer_1/BiasAdd?
Dense_Layer_1/ReluReluDense_Layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Dense_Layer_1/Relu?
!Logit_Probs/MatMul/ReadVariableOpReadVariableOp*logit_probs_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02#
!Logit_Probs/MatMul/ReadVariableOp?
Logit_Probs/MatMulMatMul Dense_Layer_1/Relu:activations:0)Logit_Probs/MatMul/ReadVariableOp:value:0*
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
Logit_Probs/BiasAdd?
IdentityIdentityLogit_Probs/BiasAdd:output:0;^batch_normalization_10/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_11/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:??????????::::::::::::::::2x
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp:P L
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
: 
?$
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_5193551

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
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
: 
?
i
M__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_5193599

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?$
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_5193756

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:?????????@
 
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
: 
?$
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_5193391

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
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
: 
?
i
M__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_5193439

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
J__inference_Dense_Layer_1_layer_call_and_return_conditional_losses_5193866

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
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
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_5194768

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity?t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????@:::::W S
/
_output_shapes
:?????????@
 
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
: 
?
?
8__inference_batch_normalization_11_layer_call_fn_5194706

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_51935512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
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
: 
?
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_5194606

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity?t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? :::::W S
/
_output_shapes
:????????? 
 
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
: 
?
N
2__inference_max_pooling2d_55_layer_call_fn_5193605

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_51935992
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
N
2__inference_max_pooling2d_54_layer_call_fn_5193445

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_51934392
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_16_layer_call_and_return_conditional_losses_5193842

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?X
?
"__inference__wrapped_model_5193285	
input2
.conv4_conv2d_54_conv2d_readvariableop_resource3
/conv4_conv2d_54_biasadd_readvariableop_resource8
4conv4_batch_normalization_10_readvariableop_resource:
6conv4_batch_normalization_10_readvariableop_1_resourceI
Econv4_batch_normalization_10_fusedbatchnormv3_readvariableop_resourceK
Gconv4_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource2
.conv4_conv2d_55_conv2d_readvariableop_resource3
/conv4_conv2d_55_biasadd_readvariableop_resource8
4conv4_batch_normalization_11_readvariableop_resource:
6conv4_batch_normalization_11_readvariableop_1_resourceI
Econv4_batch_normalization_11_fusedbatchnormv3_readvariableop_resourceK
Gconv4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource6
2conv4_dense_layer_1_matmul_readvariableop_resource7
3conv4_dense_layer_1_biasadd_readvariableop_resource4
0conv4_logit_probs_matmul_readvariableop_resource5
1conv4_logit_probs_biasadd_readvariableop_resource
identity?e
Conv4/reshape_38/ShapeShapeinput*
T0*
_output_shapes
:2
Conv4/reshape_38/Shape?
$Conv4/reshape_38/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Conv4/reshape_38/strided_slice/stack?
&Conv4/reshape_38/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Conv4/reshape_38/strided_slice/stack_1?
&Conv4/reshape_38/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Conv4/reshape_38/strided_slice/stack_2?
Conv4/reshape_38/strided_sliceStridedSliceConv4/reshape_38/Shape:output:0-Conv4/reshape_38/strided_slice/stack:output:0/Conv4/reshape_38/strided_slice/stack_1:output:0/Conv4/reshape_38/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
Conv4/reshape_38/strided_slice?
 Conv4/reshape_38/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 Conv4/reshape_38/Reshape/shape/1?
 Conv4/reshape_38/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2"
 Conv4/reshape_38/Reshape/shape/2?
 Conv4/reshape_38/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2"
 Conv4/reshape_38/Reshape/shape/3?
Conv4/reshape_38/Reshape/shapePack'Conv4/reshape_38/strided_slice:output:0)Conv4/reshape_38/Reshape/shape/1:output:0)Conv4/reshape_38/Reshape/shape/2:output:0)Conv4/reshape_38/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2 
Conv4/reshape_38/Reshape/shape?
Conv4/reshape_38/ReshapeReshapeinput'Conv4/reshape_38/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
Conv4/reshape_38/Reshape?
%Conv4/conv2d_54/Conv2D/ReadVariableOpReadVariableOp.conv4_conv2d_54_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02'
%Conv4/conv2d_54/Conv2D/ReadVariableOp?
Conv4/conv2d_54/Conv2DConv2D!Conv4/reshape_38/Reshape:output:0-Conv4/conv2d_54/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Conv4/conv2d_54/Conv2D?
&Conv4/conv2d_54/BiasAdd/ReadVariableOpReadVariableOp/conv4_conv2d_54_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&Conv4/conv2d_54/BiasAdd/ReadVariableOp?
Conv4/conv2d_54/BiasAddBiasAddConv4/conv2d_54/Conv2D:output:0.Conv4/conv2d_54/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
Conv4/conv2d_54/BiasAdd?
Conv4/conv2d_54/ReluRelu Conv4/conv2d_54/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Conv4/conv2d_54/Relu?
+Conv4/batch_normalization_10/ReadVariableOpReadVariableOp4conv4_batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype02-
+Conv4/batch_normalization_10/ReadVariableOp?
-Conv4/batch_normalization_10/ReadVariableOp_1ReadVariableOp6conv4_batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype02/
-Conv4/batch_normalization_10/ReadVariableOp_1?
<Conv4/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpEconv4_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02>
<Conv4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp?
>Conv4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGconv4_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02@
>Conv4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?
-Conv4/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3"Conv4/conv2d_54/Relu:activations:03Conv4/batch_normalization_10/ReadVariableOp:value:05Conv4/batch_normalization_10/ReadVariableOp_1:value:0DConv4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0FConv4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( 2/
-Conv4/batch_normalization_10/FusedBatchNormV3?
Conv4/max_pooling2d_54/MaxPoolMaxPool1Conv4/batch_normalization_10/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2 
Conv4/max_pooling2d_54/MaxPool?
%Conv4/conv2d_55/Conv2D/ReadVariableOpReadVariableOp.conv4_conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02'
%Conv4/conv2d_55/Conv2D/ReadVariableOp?
Conv4/conv2d_55/Conv2DConv2D'Conv4/max_pooling2d_54/MaxPool:output:0-Conv4/conv2d_55/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv4/conv2d_55/Conv2D?
&Conv4/conv2d_55/BiasAdd/ReadVariableOpReadVariableOp/conv4_conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&Conv4/conv2d_55/BiasAdd/ReadVariableOp?
Conv4/conv2d_55/BiasAddBiasAddConv4/conv2d_55/Conv2D:output:0.Conv4/conv2d_55/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
Conv4/conv2d_55/BiasAdd?
Conv4/conv2d_55/ReluRelu Conv4/conv2d_55/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Conv4/conv2d_55/Relu?
+Conv4/batch_normalization_11/ReadVariableOpReadVariableOp4conv4_batch_normalization_11_readvariableop_resource*
_output_shapes
:@*
dtype02-
+Conv4/batch_normalization_11/ReadVariableOp?
-Conv4/batch_normalization_11/ReadVariableOp_1ReadVariableOp6conv4_batch_normalization_11_readvariableop_1_resource*
_output_shapes
:@*
dtype02/
-Conv4/batch_normalization_11/ReadVariableOp_1?
<Conv4/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpEconv4_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02>
<Conv4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp?
>Conv4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGconv4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02@
>Conv4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?
-Conv4/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3"Conv4/conv2d_55/Relu:activations:03Conv4/batch_normalization_11/ReadVariableOp:value:05Conv4/batch_normalization_11/ReadVariableOp_1:value:0DConv4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0FConv4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2/
-Conv4/batch_normalization_11/FusedBatchNormV3?
Conv4/max_pooling2d_55/MaxPoolMaxPool1Conv4/batch_normalization_11/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2 
Conv4/max_pooling2d_55/MaxPool?
Conv4/flatten_38/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
Conv4/flatten_38/Const?
Conv4/flatten_38/ReshapeReshape'Conv4/max_pooling2d_55/MaxPool:output:0Conv4/flatten_38/Const:output:0*
T0*(
_output_shapes
:??????????2
Conv4/flatten_38/Reshape?
Conv4/dropout_16/IdentityIdentity!Conv4/flatten_38/Reshape:output:0*
T0*(
_output_shapes
:??????????2
Conv4/dropout_16/Identity?
)Conv4/Dense_Layer_1/MatMul/ReadVariableOpReadVariableOp2conv4_dense_layer_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02+
)Conv4/Dense_Layer_1/MatMul/ReadVariableOp?
Conv4/Dense_Layer_1/MatMulMatMul"Conv4/dropout_16/Identity:output:01Conv4/Dense_Layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
Conv4/Dense_Layer_1/MatMul?
*Conv4/Dense_Layer_1/BiasAdd/ReadVariableOpReadVariableOp3conv4_dense_layer_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02,
*Conv4/Dense_Layer_1/BiasAdd/ReadVariableOp?
Conv4/Dense_Layer_1/BiasAddBiasAdd$Conv4/Dense_Layer_1/MatMul:product:02Conv4/Dense_Layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
Conv4/Dense_Layer_1/BiasAdd?
Conv4/Dense_Layer_1/ReluRelu$Conv4/Dense_Layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Conv4/Dense_Layer_1/Relu?
'Conv4/Logit_Probs/MatMul/ReadVariableOpReadVariableOp0conv4_logit_probs_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02)
'Conv4/Logit_Probs/MatMul/ReadVariableOp?
Conv4/Logit_Probs/MatMulMatMul&Conv4/Dense_Layer_1/Relu:activations:0/Conv4/Logit_Probs/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Conv4/Logit_Probs/MatMul?
(Conv4/Logit_Probs/BiasAdd/ReadVariableOpReadVariableOp1conv4_logit_probs_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02*
(Conv4/Logit_Probs/BiasAdd/ReadVariableOp?
Conv4/Logit_Probs/BiasAddBiasAdd"Conv4/Logit_Probs/MatMul:product:00Conv4/Logit_Probs/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Conv4/Logit_Probs/BiasAddv
IdentityIdentity"Conv4/Logit_Probs/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:??????????:::::::::::::::::O K
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
: 
?
e
,__inference_dropout_16_layer_call_fn_5194827

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dropout_16_layer_call_and_return_conditional_losses_51938372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
G__inference_dropout_16_layer_call_and_return_conditional_losses_5193837

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
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
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
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_5193892

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
8__inference_batch_normalization_10_layer_call_fn_5194619

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_51936662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
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
: 
?7
?
B__inference_Conv4_layer_call_and_return_conditional_losses_5194090

inputs
conv2d_54_5194047
conv2d_54_5194049"
batch_normalization_10_5194052"
batch_normalization_10_5194054"
batch_normalization_10_5194056"
batch_normalization_10_5194058
conv2d_55_5194062
conv2d_55_5194064"
batch_normalization_11_5194067"
batch_normalization_11_5194069"
batch_normalization_11_5194071"
batch_normalization_11_5194073
dense_layer_1_5194079
dense_layer_1_5194081
logit_probs_5194084
logit_probs_5194086
identity??%Dense_Layer_1/StatefulPartitionedCall?#Logit_Probs/StatefulPartitionedCall?.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?!conv2d_54/StatefulPartitionedCall?!conv2d_55/StatefulPartitionedCall?
reshape_38/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_reshape_38_layer_call_and_return_conditional_losses_51936232
reshape_38/PartitionedCall?
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCall#reshape_38/PartitionedCall:output:0conv2d_54_5194047conv2d_54_5194049*
Tin
2*
Tout
2*/
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_54_layer_call_and_return_conditional_losses_51932972#
!conv2d_54/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0batch_normalization_10_5194052batch_normalization_10_5194054batch_normalization_10_5194056batch_normalization_10_5194058*
Tin	
2*
Tout
2*/
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_519368420
.batch_normalization_10/StatefulPartitionedCall?
 max_pooling2d_54/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_51934392"
 max_pooling2d_54/PartitionedCall?
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_54/PartitionedCall:output:0conv2d_55_5194062conv2d_55_5194064*
Tin
2*
Tout
2*/
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_55_layer_call_and_return_conditional_losses_51934572#
!conv2d_55/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0batch_normalization_11_5194067batch_normalization_11_5194069batch_normalization_11_5194071batch_normalization_11_5194073*
Tin	
2*
Tout
2*/
_output_shapes
:?????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_519377420
.batch_normalization_11/StatefulPartitionedCall?
 max_pooling2d_55/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_51935992"
 max_pooling2d_55/PartitionedCall?
flatten_38/PartitionedCallPartitionedCall)max_pooling2d_55/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_flatten_38_layer_call_and_return_conditional_losses_51938172
flatten_38/PartitionedCall?
dropout_16/PartitionedCallPartitionedCall#flatten_38/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dropout_16_layer_call_and_return_conditional_losses_51938422
dropout_16/PartitionedCall?
%Dense_Layer_1/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0dense_layer_1_5194079dense_layer_1_5194081*
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
CPU

GPU2*0J 8*S
fNRL
J__inference_Dense_Layer_1_layer_call_and_return_conditional_losses_51938662'
%Dense_Layer_1/StatefulPartitionedCall?
#Logit_Probs/StatefulPartitionedCallStatefulPartitionedCall.Dense_Layer_1/StatefulPartitionedCall:output:0logit_probs_5194084logit_probs_5194086*
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
CPU

GPU2*0J 8*Q
fLRJ
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_51938922%
#Logit_Probs/StatefulPartitionedCall?
IdentityIdentity,Logit_Probs/StatefulPartitionedCall:output:0&^Dense_Layer_1/StatefulPartitionedCall$^Logit_Probs/StatefulPartitionedCall/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:??????????::::::::::::::::2N
%Dense_Layer_1/StatefulPartitionedCall%Dense_Layer_1/StatefulPartitionedCall2J
#Logit_Probs/StatefulPartitionedCall#Logit_Probs/StatefulPartitionedCall2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall:P L
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
: 
?$
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_5194513

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
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
: 
?8
?
B__inference_Conv4_layer_call_and_return_conditional_losses_5193909	
input
conv2d_54_5193631
conv2d_54_5193633"
batch_normalization_10_5193711"
batch_normalization_10_5193713"
batch_normalization_10_5193715"
batch_normalization_10_5193717
conv2d_55_5193721
conv2d_55_5193723"
batch_normalization_11_5193801"
batch_normalization_11_5193803"
batch_normalization_11_5193805"
batch_normalization_11_5193807
dense_layer_1_5193877
dense_layer_1_5193879
logit_probs_5193903
logit_probs_5193905
identity??%Dense_Layer_1/StatefulPartitionedCall?#Logit_Probs/StatefulPartitionedCall?.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?!conv2d_54/StatefulPartitionedCall?!conv2d_55/StatefulPartitionedCall?"dropout_16/StatefulPartitionedCall?
reshape_38/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*/
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_reshape_38_layer_call_and_return_conditional_losses_51936232
reshape_38/PartitionedCall?
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCall#reshape_38/PartitionedCall:output:0conv2d_54_5193631conv2d_54_5193633*
Tin
2*
Tout
2*/
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_54_layer_call_and_return_conditional_losses_51932972#
!conv2d_54/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0batch_normalization_10_5193711batch_normalization_10_5193713batch_normalization_10_5193715batch_normalization_10_5193717*
Tin	
2*
Tout
2*/
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_519366620
.batch_normalization_10/StatefulPartitionedCall?
 max_pooling2d_54/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_51934392"
 max_pooling2d_54/PartitionedCall?
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_54/PartitionedCall:output:0conv2d_55_5193721conv2d_55_5193723*
Tin
2*
Tout
2*/
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_55_layer_call_and_return_conditional_losses_51934572#
!conv2d_55/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0batch_normalization_11_5193801batch_normalization_11_5193803batch_normalization_11_5193805batch_normalization_11_5193807*
Tin	
2*
Tout
2*/
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_519375620
.batch_normalization_11/StatefulPartitionedCall?
 max_pooling2d_55/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_51935992"
 max_pooling2d_55/PartitionedCall?
flatten_38/PartitionedCallPartitionedCall)max_pooling2d_55/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_flatten_38_layer_call_and_return_conditional_losses_51938172
flatten_38/PartitionedCall?
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall#flatten_38/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dropout_16_layer_call_and_return_conditional_losses_51938372$
"dropout_16/StatefulPartitionedCall?
%Dense_Layer_1/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0dense_layer_1_5193877dense_layer_1_5193879*
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
CPU

GPU2*0J 8*S
fNRL
J__inference_Dense_Layer_1_layer_call_and_return_conditional_losses_51938662'
%Dense_Layer_1/StatefulPartitionedCall?
#Logit_Probs/StatefulPartitionedCallStatefulPartitionedCall.Dense_Layer_1/StatefulPartitionedCall:output:0logit_probs_5193903logit_probs_5193905*
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
CPU

GPU2*0J 8*Q
fLRJ
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_51938922%
#Logit_Probs/StatefulPartitionedCall?
IdentityIdentity,Logit_Probs/StatefulPartitionedCall:output:0&^Dense_Layer_1/StatefulPartitionedCall$^Logit_Probs/StatefulPartitionedCall/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:??????????::::::::::::::::2N
%Dense_Layer_1/StatefulPartitionedCall%Dense_Layer_1/StatefulPartitionedCall2J
#Logit_Probs/StatefulPartitionedCall#Logit_Probs/StatefulPartitionedCall2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall:O K
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
: 
?
?
8__inference_batch_normalization_10_layer_call_fn_5194632

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_51936842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
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
: 
?
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_5193582

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity?t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@:::::i e
A
_output_shapes/
-:+???????????????????????????@
 
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
: 
?
H
,__inference_dropout_16_layer_call_fn_5194832

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dropout_16_layer_call_and_return_conditional_losses_51938422
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_10_layer_call_fn_5194544

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_51933912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
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
: 
?
c
G__inference_flatten_38_layer_call_and_return_conditional_losses_5194800

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?7
?
B__inference_Conv4_layer_call_and_return_conditional_losses_5193956	
input
conv2d_54_5193913
conv2d_54_5193915"
batch_normalization_10_5193918"
batch_normalization_10_5193920"
batch_normalization_10_5193922"
batch_normalization_10_5193924
conv2d_55_5193928
conv2d_55_5193930"
batch_normalization_11_5193933"
batch_normalization_11_5193935"
batch_normalization_11_5193937"
batch_normalization_11_5193939
dense_layer_1_5193945
dense_layer_1_5193947
logit_probs_5193950
logit_probs_5193952
identity??%Dense_Layer_1/StatefulPartitionedCall?#Logit_Probs/StatefulPartitionedCall?.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?!conv2d_54/StatefulPartitionedCall?!conv2d_55/StatefulPartitionedCall?
reshape_38/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*/
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_reshape_38_layer_call_and_return_conditional_losses_51936232
reshape_38/PartitionedCall?
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCall#reshape_38/PartitionedCall:output:0conv2d_54_5193913conv2d_54_5193915*
Tin
2*
Tout
2*/
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_54_layer_call_and_return_conditional_losses_51932972#
!conv2d_54/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0batch_normalization_10_5193918batch_normalization_10_5193920batch_normalization_10_5193922batch_normalization_10_5193924*
Tin	
2*
Tout
2*/
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_519368420
.batch_normalization_10/StatefulPartitionedCall?
 max_pooling2d_54/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_51934392"
 max_pooling2d_54/PartitionedCall?
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_54/PartitionedCall:output:0conv2d_55_5193928conv2d_55_5193930*
Tin
2*
Tout
2*/
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_55_layer_call_and_return_conditional_losses_51934572#
!conv2d_55/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0batch_normalization_11_5193933batch_normalization_11_5193935batch_normalization_11_5193937batch_normalization_11_5193939*
Tin	
2*
Tout
2*/
_output_shapes
:?????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_519377420
.batch_normalization_11/StatefulPartitionedCall?
 max_pooling2d_55/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_51935992"
 max_pooling2d_55/PartitionedCall?
flatten_38/PartitionedCallPartitionedCall)max_pooling2d_55/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_flatten_38_layer_call_and_return_conditional_losses_51938172
flatten_38/PartitionedCall?
dropout_16/PartitionedCallPartitionedCall#flatten_38/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dropout_16_layer_call_and_return_conditional_losses_51938422
dropout_16/PartitionedCall?
%Dense_Layer_1/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0dense_layer_1_5193945dense_layer_1_5193947*
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
CPU

GPU2*0J 8*S
fNRL
J__inference_Dense_Layer_1_layer_call_and_return_conditional_losses_51938662'
%Dense_Layer_1/StatefulPartitionedCall?
#Logit_Probs/StatefulPartitionedCallStatefulPartitionedCall.Dense_Layer_1/StatefulPartitionedCall:output:0logit_probs_5193950logit_probs_5193952*
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
CPU

GPU2*0J 8*Q
fLRJ
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_51938922%
#Logit_Probs/StatefulPartitionedCall?
IdentityIdentity,Logit_Probs/StatefulPartitionedCall:output:0&^Dense_Layer_1/StatefulPartitionedCall$^Logit_Probs/StatefulPartitionedCall/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:??????????::::::::::::::::2N
%Dense_Layer_1/StatefulPartitionedCall%Dense_Layer_1/StatefulPartitionedCall2J
#Logit_Probs/StatefulPartitionedCall#Logit_Probs/StatefulPartitionedCall2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall:O K
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
: 
?
?
%__inference_signature_wrapper_5194196	
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
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*'
_output_shapes
:?????????
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference__wrapped_model_51932852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:??????????::::::::::::::::22
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
: 
?
?
'__inference_Conv4_layer_call_fn_5194414

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
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_Conv4_layer_call_and_return_conditional_losses_51940062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:??????????::::::::::::::::22
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
: 
?
?
'__inference_Conv4_layer_call_fn_5194451

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
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*'
_output_shapes
:?????????
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_Conv4_layer_call_and_return_conditional_losses_51940902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:??????????::::::::::::::::22
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
: 
?
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_5193422

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity?t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? :::::i e
A
_output_shapes/
-:+??????????????????????????? 
 
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
: 
?$
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_5193666

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:????????? 
 
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
: 
?
?
+__inference_conv2d_54_layer_call_fn_5193307

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_54_layer_call_and_return_conditional_losses_51932972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
H
,__inference_flatten_38_layer_call_fn_5194805

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_flatten_38_layer_call_and_return_conditional_losses_51938172
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_5194531

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity?t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? :::::i e
A
_output_shapes/
-:+??????????????????????????? 
 
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
: 
?
?
8__inference_batch_normalization_11_layer_call_fn_5194781

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_51937562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
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
: 
?
?
8__inference_batch_normalization_10_layer_call_fn_5194557

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_51934222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
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
: 
?
?
J__inference_Dense_Layer_1_layer_call_and_return_conditional_losses_5194843

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
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
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?r
?
 __inference__traced_save_5195051
file_prefix/
+savev2_conv2d_54_kernel_read_readvariableop-
)savev2_conv2d_54_bias_read_readvariableop;
7savev2_batch_normalization_10_gamma_read_readvariableop:
6savev2_batch_normalization_10_beta_read_readvariableopA
=savev2_batch_normalization_10_moving_mean_read_readvariableopE
Asavev2_batch_normalization_10_moving_variance_read_readvariableop/
+savev2_conv2d_55_kernel_read_readvariableop-
)savev2_conv2d_55_bias_read_readvariableop;
7savev2_batch_normalization_11_gamma_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableop6
2savev2_dense_layer_1_38_kernel_read_readvariableop4
0savev2_dense_layer_1_38_bias_read_readvariableop4
0savev2_logit_probs_38_kernel_read_readvariableop2
.savev2_logit_probs_38_bias_read_readvariableop(
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
"savev2_count_2_read_readvariableop6
2savev2_adam_conv2d_54_kernel_m_read_readvariableop4
0savev2_adam_conv2d_54_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_10_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_10_beta_m_read_readvariableop6
2savev2_adam_conv2d_55_kernel_m_read_readvariableop4
0savev2_adam_conv2d_55_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_11_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_11_beta_m_read_readvariableop=
9savev2_adam_dense_layer_1_38_kernel_m_read_readvariableop;
7savev2_adam_dense_layer_1_38_bias_m_read_readvariableop;
7savev2_adam_logit_probs_38_kernel_m_read_readvariableop9
5savev2_adam_logit_probs_38_bias_m_read_readvariableop6
2savev2_adam_conv2d_54_kernel_v_read_readvariableop4
0savev2_adam_conv2d_54_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_10_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_10_beta_v_read_readvariableop6
2savev2_adam_conv2d_55_kernel_v_read_readvariableop4
0savev2_adam_conv2d_55_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_11_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_11_beta_v_read_readvariableop=
9savev2_adam_dense_layer_1_38_kernel_v_read_readvariableop;
7savev2_adam_dense_layer_1_38_bias_v_read_readvariableop;
7savev2_adam_logit_probs_38_kernel_v_read_readvariableop9
5savev2_adam_logit_probs_38_bias_v_read_readvariableop
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
value3B1 B+_temp_74d2c8effb44482abc6e912f8c4f3370/part2	
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*?
value?B?3B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_54_kernel_read_readvariableop)savev2_conv2d_54_bias_read_readvariableop7savev2_batch_normalization_10_gamma_read_readvariableop6savev2_batch_normalization_10_beta_read_readvariableop=savev2_batch_normalization_10_moving_mean_read_readvariableopAsavev2_batch_normalization_10_moving_variance_read_readvariableop+savev2_conv2d_55_kernel_read_readvariableop)savev2_conv2d_55_bias_read_readvariableop7savev2_batch_normalization_11_gamma_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop2savev2_dense_layer_1_38_kernel_read_readvariableop0savev2_dense_layer_1_38_bias_read_readvariableop0savev2_logit_probs_38_kernel_read_readvariableop.savev2_logit_probs_38_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop2savev2_adam_conv2d_54_kernel_m_read_readvariableop0savev2_adam_conv2d_54_bias_m_read_readvariableop>savev2_adam_batch_normalization_10_gamma_m_read_readvariableop=savev2_adam_batch_normalization_10_beta_m_read_readvariableop2savev2_adam_conv2d_55_kernel_m_read_readvariableop0savev2_adam_conv2d_55_bias_m_read_readvariableop>savev2_adam_batch_normalization_11_gamma_m_read_readvariableop=savev2_adam_batch_normalization_11_beta_m_read_readvariableop9savev2_adam_dense_layer_1_38_kernel_m_read_readvariableop7savev2_adam_dense_layer_1_38_bias_m_read_readvariableop7savev2_adam_logit_probs_38_kernel_m_read_readvariableop5savev2_adam_logit_probs_38_bias_m_read_readvariableop2savev2_adam_conv2d_54_kernel_v_read_readvariableop0savev2_adam_conv2d_54_bias_v_read_readvariableop>savev2_adam_batch_normalization_10_gamma_v_read_readvariableop=savev2_adam_batch_normalization_10_beta_v_read_readvariableop2savev2_adam_conv2d_55_kernel_v_read_readvariableop0savev2_adam_conv2d_55_bias_v_read_readvariableop>savev2_adam_batch_normalization_11_gamma_v_read_readvariableop=savev2_adam_batch_normalization_11_beta_v_read_readvariableop9savev2_adam_dense_layer_1_38_kernel_v_read_readvariableop7savev2_adam_dense_layer_1_38_bias_v_read_readvariableop7savev2_adam_logit_probs_38_kernel_v_read_readvariableop5savev2_adam_logit_probs_38_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *A
dtypes7
523	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : : : @:@:@:@:@:@:	?d:d:d
:
: : : : : : : : : : : : : : : : @:@:@:@:	?d:d:d
:
: : : : : @:@:@:@:	?d:d:d
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:%!

_output_shapes
:	?d: 

_output_shapes
:d:$ 

_output_shapes

:d
: 

_output_shapes
:
:
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
: :,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :, (
&
_output_shapes
: @: !

_output_shapes
:@: "

_output_shapes
:@: #

_output_shapes
:@:%$!

_output_shapes
:	?d: %

_output_shapes
:d:$& 

_output_shapes

:d
: '

_output_shapes
:
:,((
&
_output_shapes
: : )

_output_shapes
: : *

_output_shapes
: : +

_output_shapes
: :,,(
&
_output_shapes
: @: -

_output_shapes
:@: .

_output_shapes
:@: /

_output_shapes
:@:%0!

_output_shapes
:	?d: 1

_output_shapes
:d:$2 

_output_shapes

:d
: 3

_output_shapes
:
:4

_output_shapes
: 
?

?
F__inference_conv2d_54_layer_call_and_return_conditional_losses_5193297

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????:::i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
c
G__inference_reshape_38_layer_call_and_return_conditional_losses_5194465

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
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
tensorflow/serving/predict:??
?e
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"?a
_tf_keras_model?a{"class_name": "Model", "name": "Conv4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Conv4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}, "name": "Input", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape_38", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [28, 28, 1]}}, "name": "reshape_38", "inbound_nodes": [[["Input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_54", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_54", "inbound_nodes": [[["reshape_38", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_10", "inbound_nodes": [[["conv2d_54", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_54", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_54", "inbound_nodes": [[["batch_normalization_10", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_55", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_55", "inbound_nodes": [[["max_pooling2d_54", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_11", "inbound_nodes": [[["conv2d_55", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_55", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_55", "inbound_nodes": [[["batch_normalization_11", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_38", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_38", "inbound_nodes": [[["max_pooling2d_55", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_16", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_16", "inbound_nodes": [[["flatten_38", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_Layer_1", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_Layer_1", "inbound_nodes": [[["dropout_16", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Logit_Probs", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Logit_Probs", "inbound_nodes": [[["Dense_Layer_1", 0, 0, {}]]]}], "input_layers": [["Input", 0, 0]], "output_layers": [["Logit_Probs", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "Conv4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}, "name": "Input", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape_38", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [28, 28, 1]}}, "name": "reshape_38", "inbound_nodes": [[["Input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_54", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_54", "inbound_nodes": [[["reshape_38", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_10", "inbound_nodes": [[["conv2d_54", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_54", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_54", "inbound_nodes": [[["batch_normalization_10", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_55", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_55", "inbound_nodes": [[["max_pooling2d_54", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_11", "inbound_nodes": [[["conv2d_55", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_55", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_55", "inbound_nodes": [[["batch_normalization_11", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_38", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_38", "inbound_nodes": [[["max_pooling2d_55", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_16", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_16", "inbound_nodes": [[["flatten_38", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_Layer_1", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_Layer_1", "inbound_nodes": [[["dropout_16", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Logit_Probs", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Logit_Probs", "inbound_nodes": [[["Dense_Layer_1", 0, 0, {}]]]}], "input_layers": [["Input", 0, 0]], "output_layers": [["Logit_Probs", 0, 0]]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}}, "metrics": ["accuracy", {"class_name": "SparseTopKCategoricalAccuracy", "config": {"name": "sparse_top_k_categorical_accuracy", "dtype": "float32", "k": 3}}], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "Input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}}
?
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "reshape_38", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [28, 28, 1]}}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_54", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "stateful": false, "config": {"name": "conv2d_54", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}
?	
axis
	gamma
beta
 moving_mean
!moving_variance
"trainable_variables
#	variables
$regularization_losses
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 26, 32]}}
?
&trainable_variables
'	variables
(regularization_losses
)	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_54", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling2d_54", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


*kernel
+bias
,trainable_variables
-	variables
.regularization_losses
/	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_55", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_55", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 13, 32]}}
?	
0axis
	1gamma
2beta
3moving_mean
4moving_variance
5trainable_variables
6	variables
7regularization_losses
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 11, 64]}}
?
9trainable_variables
:	variables
;regularization_losses
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_55", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling2d_55", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
=trainable_variables
>	variables
?regularization_losses
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten_38", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_16", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?

Ekernel
Fbias
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "Dense_Layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Dense_Layer_1", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1600}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1600]}}
?

Kkernel
Lbias
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "Logit_Probs", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Logit_Probs", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?
Qiter

Rbeta_1

Sbeta_2
	Tdecay
Ulearning_ratem?m?m?m?*m?+m?1m?2m?Em?Fm?Km?Lm?v?v?v?v?*v?+v?1v?2v?Ev?Fv?Kv?Lv?"
	optimizer
v
0
1
2
3
*4
+5
16
27
E8
F9
K10
L11"
trackable_list_wrapper
?
0
1
2
3
 4
!5
*6
+7
18
29
310
411
E12
F13
K14
L15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Vmetrics
Wlayer_regularization_losses
trainable_variables

Xlayers
Ynon_trainable_variables
	variables
Zlayer_metrics
regularization_losses
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
[metrics
\layer_regularization_losses

]layers
^non_trainable_variables
trainable_variables
	variables
_layer_metrics
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_54/kernel
: 2conv2d_54/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
`metrics
alayer_regularization_losses

blayers
cnon_trainable_variables
trainable_variables
	variables
dlayer_metrics
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_10/gamma
):' 2batch_normalization_10/beta
2:0  (2"batch_normalization_10/moving_mean
6:4  (2&batch_normalization_10/moving_variance
.
0
1"
trackable_list_wrapper
<
0
1
 2
!3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
emetrics
flayer_regularization_losses

glayers
hnon_trainable_variables
"trainable_variables
#	variables
ilayer_metrics
$regularization_losses
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
jmetrics
klayer_regularization_losses

llayers
mnon_trainable_variables
&trainable_variables
'	variables
nlayer_metrics
(regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_55/kernel
:@2conv2d_55/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
ometrics
player_regularization_losses

qlayers
rnon_trainable_variables
,trainable_variables
-	variables
slayer_metrics
.regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_11/gamma
):'@2batch_normalization_11/beta
2:0@ (2"batch_normalization_11/moving_mean
6:4@ (2&batch_normalization_11/moving_variance
.
10
21"
trackable_list_wrapper
<
10
21
32
43"
trackable_list_wrapper
 "
trackable_list_wrapper
?
tmetrics
ulayer_regularization_losses

vlayers
wnon_trainable_variables
5trainable_variables
6	variables
xlayer_metrics
7regularization_losses
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
ymetrics
zlayer_regularization_losses

{layers
|non_trainable_variables
9trainable_variables
:	variables
}layer_metrics
;regularization_losses
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
~metrics
layer_regularization_losses
?layers
?non_trainable_variables
=trainable_variables
>	variables
?layer_metrics
?regularization_losses
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
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
Atrainable_variables
B	variables
?layer_metrics
Cregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(	?d2Dense_Layer_1_38/kernel
#:!d2Dense_Layer_1_38/bias
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
Gtrainable_variables
H	variables
?layer_metrics
Iregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%d
2Logit_Probs_38/kernel
!:
2Logit_Probs_38/bias
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
Mtrainable_variables
N	variables
?layer_metrics
Oregularization_losses
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
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
v
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
11"
trackable_list_wrapper
<
 0
!1
32
43"
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
.
 0
!1"
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
.
30
41"
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
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "SparseTopKCategoricalAccuracy", "name": "sparse_top_k_categorical_accuracy", "dtype": "float32", "config": {"name": "sparse_top_k_categorical_accuracy", "dtype": "float32", "k": 3}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
/:- 2Adam/conv2d_54/kernel/m
!: 2Adam/conv2d_54/bias/m
/:- 2#Adam/batch_normalization_10/gamma/m
.:, 2"Adam/batch_normalization_10/beta/m
/:- @2Adam/conv2d_55/kernel/m
!:@2Adam/conv2d_55/bias/m
/:-@2#Adam/batch_normalization_11/gamma/m
.:,@2"Adam/batch_normalization_11/beta/m
/:-	?d2Adam/Dense_Layer_1_38/kernel/m
(:&d2Adam/Dense_Layer_1_38/bias/m
,:*d
2Adam/Logit_Probs_38/kernel/m
&:$
2Adam/Logit_Probs_38/bias/m
/:- 2Adam/conv2d_54/kernel/v
!: 2Adam/conv2d_54/bias/v
/:- 2#Adam/batch_normalization_10/gamma/v
.:, 2"Adam/batch_normalization_10/beta/v
/:- @2Adam/conv2d_55/kernel/v
!:@2Adam/conv2d_55/bias/v
/:-@2#Adam/batch_normalization_11/gamma/v
.:,@2"Adam/batch_normalization_11/beta/v
/:-	?d2Adam/Dense_Layer_1_38/kernel/v
(:&d2Adam/Dense_Layer_1_38/bias/v
,:*d
2Adam/Logit_Probs_38/kernel/v
&:$
2Adam/Logit_Probs_38/bias/v
?2?
"__inference__wrapped_model_5193285?
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
?2?
'__inference_Conv4_layer_call_fn_5194451
'__inference_Conv4_layer_call_fn_5194041
'__inference_Conv4_layer_call_fn_5194414
'__inference_Conv4_layer_call_fn_5194125?
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
?2?
B__inference_Conv4_layer_call_and_return_conditional_losses_5194377
B__inference_Conv4_layer_call_and_return_conditional_losses_5193909
B__inference_Conv4_layer_call_and_return_conditional_losses_5193956
B__inference_Conv4_layer_call_and_return_conditional_losses_5194303?
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
,__inference_reshape_38_layer_call_fn_5194470?
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
G__inference_reshape_38_layer_call_and_return_conditional_losses_5194465?
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
+__inference_conv2d_54_layer_call_fn_5193307?
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
annotations? *7?4
2?/+???????????????????????????
?2?
F__inference_conv2d_54_layer_call_and_return_conditional_losses_5193297?
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
annotations? *7?4
2?/+???????????????????????????
?2?
8__inference_batch_normalization_10_layer_call_fn_5194619
8__inference_batch_normalization_10_layer_call_fn_5194557
8__inference_batch_normalization_10_layer_call_fn_5194632
8__inference_batch_normalization_10_layer_call_fn_5194544?
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
?2?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_5194531
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_5194513
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_5194588
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_5194606?
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
2__inference_max_pooling2d_54_layer_call_fn_5193445?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
M__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_5193439?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
+__inference_conv2d_55_layer_call_fn_5193467?
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
annotations? *7?4
2?/+??????????????????????????? 
?2?
F__inference_conv2d_55_layer_call_and_return_conditional_losses_5193457?
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
annotations? *7?4
2?/+??????????????????????????? 
?2?
8__inference_batch_normalization_11_layer_call_fn_5194794
8__inference_batch_normalization_11_layer_call_fn_5194706
8__inference_batch_normalization_11_layer_call_fn_5194719
8__inference_batch_normalization_11_layer_call_fn_5194781?
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
?2?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_5194675
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_5194693
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_5194768
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_5194750?
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
2__inference_max_pooling2d_55_layer_call_fn_5193605?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
M__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_5193599?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
,__inference_flatten_38_layer_call_fn_5194805?
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
G__inference_flatten_38_layer_call_and_return_conditional_losses_5194800?
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
,__inference_dropout_16_layer_call_fn_5194827
,__inference_dropout_16_layer_call_fn_5194832?
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
G__inference_dropout_16_layer_call_and_return_conditional_losses_5194822
G__inference_dropout_16_layer_call_and_return_conditional_losses_5194817?
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
/__inference_Dense_Layer_1_layer_call_fn_5194852?
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
J__inference_Dense_Layer_1_layer_call_and_return_conditional_losses_5194843?
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
-__inference_Logit_Probs_layer_call_fn_5194871?
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
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_5194862?
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
%__inference_signature_wrapper_5194196Input?
B__inference_Conv4_layer_call_and_return_conditional_losses_5193909r !*+1234EFKL7?4
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
B__inference_Conv4_layer_call_and_return_conditional_losses_5193956r !*+1234EFKL7?4
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
B__inference_Conv4_layer_call_and_return_conditional_losses_5194303s !*+1234EFKL8?5
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
B__inference_Conv4_layer_call_and_return_conditional_losses_5194377s !*+1234EFKL8?5
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
'__inference_Conv4_layer_call_fn_5194041e !*+1234EFKL7?4
-?*
 ?
Input??????????
p

 
? "??????????
?
'__inference_Conv4_layer_call_fn_5194125e !*+1234EFKL7?4
-?*
 ?
Input??????????
p 

 
? "??????????
?
'__inference_Conv4_layer_call_fn_5194414f !*+1234EFKL8?5
.?+
!?
inputs??????????
p

 
? "??????????
?
'__inference_Conv4_layer_call_fn_5194451f !*+1234EFKL8?5
.?+
!?
inputs??????????
p 

 
? "??????????
?
J__inference_Dense_Layer_1_layer_call_and_return_conditional_losses_5194843]EF0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????d
? ?
/__inference_Dense_Layer_1_layer_call_fn_5194852PEF0?-
&?#
!?
inputs??????????
? "??????????d?
H__inference_Logit_Probs_layer_call_and_return_conditional_losses_5194862\KL/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????

? ?
-__inference_Logit_Probs_layer_call_fn_5194871OKL/?,
%?"
 ?
inputs?????????d
? "??????????
?
"__inference__wrapped_model_5193285~ !*+1234EFKL/?,
%?"
 ?
Input??????????
? "9?6
4
Logit_Probs%?"
Logit_Probs?????????
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_5194513? !M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_5194531? !M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_5194588r !;?8
1?.
(?%
inputs????????? 
p
? "-?*
#? 
0????????? 
? ?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_5194606r !;?8
1?.
(?%
inputs????????? 
p 
? "-?*
#? 
0????????? 
? ?
8__inference_batch_normalization_10_layer_call_fn_5194544? !M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
8__inference_batch_normalization_10_layer_call_fn_5194557? !M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
8__inference_batch_normalization_10_layer_call_fn_5194619e !;?8
1?.
(?%
inputs????????? 
p
? " ?????????? ?
8__inference_batch_normalization_10_layer_call_fn_5194632e !;?8
1?.
(?%
inputs????????? 
p 
? " ?????????? ?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_5194675?1234M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_5194693?1234M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_5194750r1234;?8
1?.
(?%
inputs?????????@
p
? "-?*
#? 
0?????????@
? ?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_5194768r1234;?8
1?.
(?%
inputs?????????@
p 
? "-?*
#? 
0?????????@
? ?
8__inference_batch_normalization_11_layer_call_fn_5194706?1234M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
8__inference_batch_normalization_11_layer_call_fn_5194719?1234M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
8__inference_batch_normalization_11_layer_call_fn_5194781e1234;?8
1?.
(?%
inputs?????????@
p
? " ??????????@?
8__inference_batch_normalization_11_layer_call_fn_5194794e1234;?8
1?.
(?%
inputs?????????@
p 
? " ??????????@?
F__inference_conv2d_54_layer_call_and_return_conditional_losses_5193297?I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+??????????????????????????? 
? ?
+__inference_conv2d_54_layer_call_fn_5193307?I?F
??<
:?7
inputs+???????????????????????????
? "2?/+??????????????????????????? ?
F__inference_conv2d_55_layer_call_and_return_conditional_losses_5193457?*+I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????@
? ?
+__inference_conv2d_55_layer_call_fn_5193467?*+I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+???????????????????????????@?
G__inference_dropout_16_layer_call_and_return_conditional_losses_5194817^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
G__inference_dropout_16_layer_call_and_return_conditional_losses_5194822^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
,__inference_dropout_16_layer_call_fn_5194827Q4?1
*?'
!?
inputs??????????
p
? "????????????
,__inference_dropout_16_layer_call_fn_5194832Q4?1
*?'
!?
inputs??????????
p 
? "????????????
G__inference_flatten_38_layer_call_and_return_conditional_losses_5194800a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0??????????
? ?
,__inference_flatten_38_layer_call_fn_5194805T7?4
-?*
(?%
inputs?????????@
? "????????????
M__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_5193439?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_54_layer_call_fn_5193445?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
M__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_5193599?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_55_layer_call_fn_5193605?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_reshape_38_layer_call_and_return_conditional_losses_5194465a0?-
&?#
!?
inputs??????????
? "-?*
#? 
0?????????
? ?
,__inference_reshape_38_layer_call_fn_5194470T0?-
&?#
!?
inputs??????????
? " ???????????
%__inference_signature_wrapper_5194196? !*+1234EFKL8?5
? 
.?+
)
Input ?
Input??????????"9?6
4
Logit_Probs%?"
Logit_Probs?????????
