��
��
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
dtypetype�
�
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
executor_typestring �
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.3.02v2.3.0-rc2-23-gb36436b0878��
�
 dense_features/embedding_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*1
shared_name" dense_features/embedding_weights
�
4dense_features/embedding_weights/Read/ReadVariableOpReadVariableOp dense_features/embedding_weights*
_output_shapes
:	�
*
dtype0
�
"dense_features_1/embedding_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��
*3
shared_name$"dense_features_1/embedding_weights
�
6dense_features_1/embedding_weights/Read/ReadVariableOpReadVariableOp"dense_features_1/embedding_weights* 
_output_shapes
:
��
*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:
*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:
*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:

*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:
*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:
*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
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
u
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:�*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:�*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:�*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:�*
dtype0
y
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*!
shared_nametrue_positives_1
r
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes	
:�*
dtype0
y
true_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*!
shared_nametrue_negatives_1
r
$true_negatives_1/Read/ReadVariableOpReadVariableOptrue_negatives_1*
_output_shapes	
:�*
dtype0
{
false_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_namefalse_positives_1
t
%false_positives_1/Read/ReadVariableOpReadVariableOpfalse_positives_1*
_output_shapes	
:�*
dtype0
{
false_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_namefalse_negatives_1
t
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes	
:�*
dtype0
�
'Adam/dense_features/embedding_weights/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*8
shared_name)'Adam/dense_features/embedding_weights/m
�
;Adam/dense_features/embedding_weights/m/Read/ReadVariableOpReadVariableOp'Adam/dense_features/embedding_weights/m*
_output_shapes
:	�
*
dtype0
�
)Adam/dense_features_1/embedding_weights/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��
*:
shared_name+)Adam/dense_features_1/embedding_weights/m
�
=Adam/dense_features_1/embedding_weights/m/Read/ReadVariableOpReadVariableOp)Adam/dense_features_1/embedding_weights/m* 
_output_shapes
:
��
*
dtype0
�
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:
*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:
*
dtype0
�
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:

*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:
*
dtype0
�
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:
*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0
�
'Adam/dense_features/embedding_weights/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*8
shared_name)'Adam/dense_features/embedding_weights/v
�
;Adam/dense_features/embedding_weights/v/Read/ReadVariableOpReadVariableOp'Adam/dense_features/embedding_weights/v*
_output_shapes
:	�
*
dtype0
�
)Adam/dense_features_1/embedding_weights/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��
*:
shared_name+)Adam/dense_features_1/embedding_weights/v
�
=Adam/dense_features_1/embedding_weights/v/Read/ReadVariableOpReadVariableOp)Adam/dense_features_1/embedding_weights/v* 
_output_shapes
:
��
*
dtype0
�
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:
*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:
*
dtype0
�
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:

*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:
*
dtype0
�
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:
*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�>
value�>B�> B�>
�
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
layer_with_weights-4
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
 
�
_feature_columns

_resources
'#movieId_embedding/embedding_weights
	variables
trainable_variables
regularization_losses
	keras_api
�
_feature_columns

_resources
&"userId_embedding/embedding_weights
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
 	keras_api
h

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
h

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
h

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
�
3iter

4beta_1

5beta_2
	6decay
7learning_ratemtmu!mv"mw'mx(my-mz.m{v|v}!v~"v'v�(v�-v�.v�
8
0
1
!2
"3
'4
(5
-6
.7
8
0
1
!2
"3
'4
(5
-6
.7
 
�

8layers

	variables
9layer_metrics
trainable_variables
regularization_losses
:layer_regularization_losses
;non_trainable_variables
<metrics
 
 
 
��
VARIABLE_VALUE dense_features/embedding_weightsTlayer_with_weights-0/movieId_embedding.Sembedding_weights/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
�

=layers
	variables
>layer_metrics
trainable_variables
regularization_losses
?layer_regularization_losses
@non_trainable_variables
Ametrics
 
 
��
VARIABLE_VALUE"dense_features_1/embedding_weightsSlayer_with_weights-1/userId_embedding.Sembedding_weights/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
�

Blayers
	variables
Clayer_metrics
trainable_variables
regularization_losses
Dlayer_regularization_losses
Enon_trainable_variables
Fmetrics
 
 
 
�

Glayers
	variables
Hlayer_metrics
trainable_variables
regularization_losses
Ilayer_regularization_losses
Jnon_trainable_variables
Kmetrics
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1

!0
"1
 
�

Llayers
#	variables
Mlayer_metrics
$trainable_variables
%regularization_losses
Nlayer_regularization_losses
Onon_trainable_variables
Pmetrics
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1

'0
(1
 
�

Qlayers
)	variables
Rlayer_metrics
*trainable_variables
+regularization_losses
Slayer_regularization_losses
Tnon_trainable_variables
Umetrics
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1

-0
.1
 
�

Vlayers
/	variables
Wlayer_metrics
0trainable_variables
1regularization_losses
Xlayer_regularization_losses
Ynon_trainable_variables
Zmetrics
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
8
0
1
2
3
4
5
6
7
 
 
 

[0
\1
]2
^3
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
	_total
	`count
a	variables
b	keras_api
D
	ctotal
	dcount
e
_fn_kwargs
f	variables
g	keras_api
p
htrue_positives
itrue_negatives
jfalse_positives
kfalse_negatives
l	variables
m	keras_api
p
ntrue_positives
otrue_negatives
pfalse_positives
qfalse_negatives
r	variables
s	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

_0
`1

a	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

c0
d1

f	variables
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

h0
i1
j2
k3

l	variables
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEtrue_negatives_1=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_positives_1>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

n0
o1
p2
q3

r	variables
��
VARIABLE_VALUE'Adam/dense_features/embedding_weights/mplayer_with_weights-0/movieId_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE)Adam/dense_features_1/embedding_weights/molayer_with_weights-1/userId_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE'Adam/dense_features/embedding_weights/vplayer_with_weights-0/movieId_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE)Adam/dense_features_1/embedding_weights/volayer_with_weights-1/userId_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
r
serving_default_movieIdPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
q
serving_default_userIdPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_movieIdserving_default_userId dense_features/embedding_weights"dense_features_1/embedding_weightsdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_212613
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename4dense_features/embedding_weights/Read/ReadVariableOp6dense_features_1/embedding_weights/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp$true_negatives_1/Read/ReadVariableOp%false_positives_1/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOp;Adam/dense_features/embedding_weights/m/Read/ReadVariableOp=Adam/dense_features_1/embedding_weights/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp;Adam/dense_features/embedding_weights/v/Read/ReadVariableOp=Adam/dense_features_1/embedding_weights/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOpConst*6
Tin/
-2+	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_213625
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename dense_features/embedding_weights"dense_features_1/embedding_weightsdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1true_positivestrue_negativesfalse_positivesfalse_negativestrue_positives_1true_negatives_1false_positives_1false_negatives_1'Adam/dense_features/embedding_weights/m)Adam/dense_features_1/embedding_weights/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/m'Adam/dense_features/embedding_weights/v)Adam/dense_features_1/embedding_weights/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/v*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_213758��
�	
�
-__inference_functional_1_layer_call_fn_212581
movieid

userid
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallmovieiduseridunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_2125622
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:���������:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	movieId:KG
#
_output_shapes
:���������
 
_user_specified_nameuserId
�
�
/__inference_dense_features_layer_call_fn_213211
features_movieid
features_userid
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallfeatures_movieidfeatures_useridunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_2120602
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*5
_input_shapes$
":���������:���������:22
StatefulPartitionedCallStatefulPartitionedCall:U Q
#
_output_shapes
:���������
*
_user_specified_namefeatures/movieId:TP
#
_output_shapes
:���������
)
_user_specified_namefeatures/userId
�
�
J__inference_dense_features_layer_call_and_return_conditional_losses_212145
features

features_1_
[movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_212105
identity��
 movieId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 movieId_embedding/ExpandDims/dim�
movieId_embedding/ExpandDims
ExpandDimsfeatures)movieId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������2
movieId_embedding/ExpandDims�
0movieId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
���������22
0movieId_embedding/to_sparse_input/ignore_value/x�
*movieId_embedding/to_sparse_input/NotEqualNotEqual%movieId_embedding/ExpandDims:output:09movieId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:���������2,
*movieId_embedding/to_sparse_input/NotEqual�
)movieId_embedding/to_sparse_input/indicesWhere.movieId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:���������2+
)movieId_embedding/to_sparse_input/indices�
(movieId_embedding/to_sparse_input/valuesGatherNd%movieId_embedding/ExpandDims:output:01movieId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:���������2*
(movieId_embedding/to_sparse_input/values�
-movieId_embedding/to_sparse_input/dense_shapeShape%movieId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2/
-movieId_embedding/to_sparse_input/dense_shape�
movieId_embedding/valuesCast1movieId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:���������2
movieId_embedding/values�
7movieId_embedding/movieId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 29
7movieId_embedding/movieId_embedding_weights/Slice/begin�
6movieId_embedding/movieId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:28
6movieId_embedding/movieId_embedding_weights/Slice/size�
1movieId_embedding/movieId_embedding_weights/SliceSlice6movieId_embedding/to_sparse_input/dense_shape:output:0@movieId_embedding/movieId_embedding_weights/Slice/begin:output:0?movieId_embedding/movieId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:23
1movieId_embedding/movieId_embedding_weights/Slice�
1movieId_embedding/movieId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1movieId_embedding/movieId_embedding_weights/Const�
0movieId_embedding/movieId_embedding_weights/ProdProd:movieId_embedding/movieId_embedding_weights/Slice:output:0:movieId_embedding/movieId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: 22
0movieId_embedding/movieId_embedding_weights/Prod�
<movieId_embedding/movieId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :2>
<movieId_embedding/movieId_embedding_weights/GatherV2/indices�
9movieId_embedding/movieId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9movieId_embedding/movieId_embedding_weights/GatherV2/axis�
4movieId_embedding/movieId_embedding_weights/GatherV2GatherV26movieId_embedding/to_sparse_input/dense_shape:output:0EmovieId_embedding/movieId_embedding_weights/GatherV2/indices:output:0BmovieId_embedding/movieId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 26
4movieId_embedding/movieId_embedding_weights/GatherV2�
2movieId_embedding/movieId_embedding_weights/Cast/xPack9movieId_embedding/movieId_embedding_weights/Prod:output:0=movieId_embedding/movieId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:24
2movieId_embedding/movieId_embedding_weights/Cast/x�
9movieId_embedding/movieId_embedding_weights/SparseReshapeSparseReshape1movieId_embedding/to_sparse_input/indices:index:06movieId_embedding/to_sparse_input/dense_shape:output:0;movieId_embedding/movieId_embedding_weights/Cast/x:output:0*-
_output_shapes
:���������:2;
9movieId_embedding/movieId_embedding_weights/SparseReshape�
BmovieId_embedding/movieId_embedding_weights/SparseReshape/IdentityIdentitymovieId_embedding/values:y:0*
T0	*#
_output_shapes
:���������2D
BmovieId_embedding/movieId_embedding_weights/SparseReshape/Identity�
:movieId_embedding/movieId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2<
:movieId_embedding/movieId_embedding_weights/GreaterEqual/y�
8movieId_embedding/movieId_embedding_weights/GreaterEqualGreaterEqualKmovieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0CmovieId_embedding/movieId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:���������2:
8movieId_embedding/movieId_embedding_weights/GreaterEqual�
1movieId_embedding/movieId_embedding_weights/WhereWhere<movieId_embedding/movieId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:���������23
1movieId_embedding/movieId_embedding_weights/Where�
9movieId_embedding/movieId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������2;
9movieId_embedding/movieId_embedding_weights/Reshape/shape�
3movieId_embedding/movieId_embedding_weights/ReshapeReshape9movieId_embedding/movieId_embedding_weights/Where:index:0BmovieId_embedding/movieId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:���������25
3movieId_embedding/movieId_embedding_weights/Reshape�
;movieId_embedding/movieId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;movieId_embedding/movieId_embedding_weights/GatherV2_1/axis�
6movieId_embedding/movieId_embedding_weights/GatherV2_1GatherV2JmovieId_embedding/movieId_embedding_weights/SparseReshape:output_indices:0<movieId_embedding/movieId_embedding_weights/Reshape:output:0DmovieId_embedding/movieId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:���������28
6movieId_embedding/movieId_embedding_weights/GatherV2_1�
;movieId_embedding/movieId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;movieId_embedding/movieId_embedding_weights/GatherV2_2/axis�
6movieId_embedding/movieId_embedding_weights/GatherV2_2GatherV2KmovieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0<movieId_embedding/movieId_embedding_weights/Reshape:output:0DmovieId_embedding/movieId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:���������28
6movieId_embedding/movieId_embedding_weights/GatherV2_2�
4movieId_embedding/movieId_embedding_weights/IdentityIdentityHmovieId_embedding/movieId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:26
4movieId_embedding/movieId_embedding_weights/Identity�
EmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 2G
EmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/Const�
SmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows?movieId_embedding/movieId_embedding_weights/GatherV2_1:output:0?movieId_embedding/movieId_embedding_weights/GatherV2_2:output:0=movieId_embedding/movieId_embedding_weights/Identity:output:0NmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:���������:���������:���������:���������2U
SmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows�
WmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2Y
WmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack�
YmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2[
YmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1�
YmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2[
YmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2�
QmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicedmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0`movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0bmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0bmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2S
QmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice�
JmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/UniqueUniquecmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:���������:���������2L
JmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique�
TmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGather[movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_212105NmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*n
_classd
b`loc:@movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/212105*'
_output_shapes
:���������
*
dtype02V
TmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup�
]movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity]movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*n
_classd
b`loc:@movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/212105*'
_output_shapes
:���������
2_
]movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity�
_movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1IdentityfmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������
2a
_movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1�
CmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparseSparseSegmentMeanhmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0PmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:idx:0ZmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:���������
2E
CmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse�
;movieId_embedding/movieId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2=
;movieId_embedding/movieId_embedding_weights/Reshape_1/shape�
5movieId_embedding/movieId_embedding_weights/Reshape_1ReshapeimovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0DmovieId_embedding/movieId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:���������27
5movieId_embedding/movieId_embedding_weights/Reshape_1�
1movieId_embedding/movieId_embedding_weights/ShapeShapeLmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:23
1movieId_embedding/movieId_embedding_weights/Shape�
?movieId_embedding/movieId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2A
?movieId_embedding/movieId_embedding_weights/strided_slice/stack�
AmovieId_embedding/movieId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
AmovieId_embedding/movieId_embedding_weights/strided_slice/stack_1�
AmovieId_embedding/movieId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
AmovieId_embedding/movieId_embedding_weights/strided_slice/stack_2�
9movieId_embedding/movieId_embedding_weights/strided_sliceStridedSlice:movieId_embedding/movieId_embedding_weights/Shape:output:0HmovieId_embedding/movieId_embedding_weights/strided_slice/stack:output:0JmovieId_embedding/movieId_embedding_weights/strided_slice/stack_1:output:0JmovieId_embedding/movieId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9movieId_embedding/movieId_embedding_weights/strided_slice�
3movieId_embedding/movieId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :25
3movieId_embedding/movieId_embedding_weights/stack/0�
1movieId_embedding/movieId_embedding_weights/stackPack<movieId_embedding/movieId_embedding_weights/stack/0:output:0BmovieId_embedding/movieId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:23
1movieId_embedding/movieId_embedding_weights/stack�
0movieId_embedding/movieId_embedding_weights/TileTile>movieId_embedding/movieId_embedding_weights/Reshape_1:output:0:movieId_embedding/movieId_embedding_weights/stack:output:0*
T0
*0
_output_shapes
:������������������22
0movieId_embedding/movieId_embedding_weights/Tile�
6movieId_embedding/movieId_embedding_weights/zeros_like	ZerosLikeLmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
28
6movieId_embedding/movieId_embedding_weights/zeros_like�
+movieId_embedding/movieId_embedding_weightsSelect9movieId_embedding/movieId_embedding_weights/Tile:output:0:movieId_embedding/movieId_embedding_weights/zeros_like:y:0LmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
2-
+movieId_embedding/movieId_embedding_weights�
2movieId_embedding/movieId_embedding_weights/Cast_1Cast6movieId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:24
2movieId_embedding/movieId_embedding_weights/Cast_1�
9movieId_embedding/movieId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 2;
9movieId_embedding/movieId_embedding_weights/Slice_1/begin�
8movieId_embedding/movieId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:2:
8movieId_embedding/movieId_embedding_weights/Slice_1/size�
3movieId_embedding/movieId_embedding_weights/Slice_1Slice6movieId_embedding/movieId_embedding_weights/Cast_1:y:0BmovieId_embedding/movieId_embedding_weights/Slice_1/begin:output:0AmovieId_embedding/movieId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:25
3movieId_embedding/movieId_embedding_weights/Slice_1�
3movieId_embedding/movieId_embedding_weights/Shape_1Shape4movieId_embedding/movieId_embedding_weights:output:0*
T0*
_output_shapes
:25
3movieId_embedding/movieId_embedding_weights/Shape_1�
9movieId_embedding/movieId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:2;
9movieId_embedding/movieId_embedding_weights/Slice_2/begin�
8movieId_embedding/movieId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
���������2:
8movieId_embedding/movieId_embedding_weights/Slice_2/size�
3movieId_embedding/movieId_embedding_weights/Slice_2Slice<movieId_embedding/movieId_embedding_weights/Shape_1:output:0BmovieId_embedding/movieId_embedding_weights/Slice_2/begin:output:0AmovieId_embedding/movieId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:25
3movieId_embedding/movieId_embedding_weights/Slice_2�
7movieId_embedding/movieId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7movieId_embedding/movieId_embedding_weights/concat/axis�
2movieId_embedding/movieId_embedding_weights/concatConcatV2<movieId_embedding/movieId_embedding_weights/Slice_1:output:0<movieId_embedding/movieId_embedding_weights/Slice_2:output:0@movieId_embedding/movieId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2movieId_embedding/movieId_embedding_weights/concat�
5movieId_embedding/movieId_embedding_weights/Reshape_2Reshape4movieId_embedding/movieId_embedding_weights:output:0;movieId_embedding/movieId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:���������
27
5movieId_embedding/movieId_embedding_weights/Reshape_2�
movieId_embedding/ShapeShape>movieId_embedding/movieId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:2
movieId_embedding/Shape�
%movieId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%movieId_embedding/strided_slice/stack�
'movieId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'movieId_embedding/strided_slice/stack_1�
'movieId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'movieId_embedding/strided_slice/stack_2�
movieId_embedding/strided_sliceStridedSlice movieId_embedding/Shape:output:0.movieId_embedding/strided_slice/stack:output:00movieId_embedding/strided_slice/stack_1:output:00movieId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
movieId_embedding/strided_slice�
!movieId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2#
!movieId_embedding/Reshape/shape/1�
movieId_embedding/Reshape/shapePack(movieId_embedding/strided_slice:output:0*movieId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2!
movieId_embedding/Reshape/shape�
movieId_embedding/ReshapeReshape>movieId_embedding/movieId_embedding_weights/Reshape_2:output:0(movieId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:���������
2
movieId_embedding/Reshapeq
concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat/concat_dim�
concat/concatIdentity"movieId_embedding/Reshape:output:0*
T0*'
_output_shapes
:���������
2
concat/concatj
IdentityIdentityconcat/concat:output:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*5
_input_shapes$
":���������:���������::M I
#
_output_shapes
:���������
"
_user_specified_name
features:MI
#
_output_shapes
:���������
"
_user_specified_name
features
�
�
A__inference_dense_layer_call_and_return_conditional_losses_213429

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������
2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
L__inference_dense_features_1_layer_call_and_return_conditional_losses_213304
features_movieid
features_userid]
Yuserid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_213264
identity��
userId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
userId_embedding/ExpandDims/dim�
userId_embedding/ExpandDims
ExpandDimsfeatures_userid(userId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������2
userId_embedding/ExpandDims�
/userId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
���������21
/userId_embedding/to_sparse_input/ignore_value/x�
)userId_embedding/to_sparse_input/NotEqualNotEqual$userId_embedding/ExpandDims:output:08userId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:���������2+
)userId_embedding/to_sparse_input/NotEqual�
(userId_embedding/to_sparse_input/indicesWhere-userId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:���������2*
(userId_embedding/to_sparse_input/indices�
'userId_embedding/to_sparse_input/valuesGatherNd$userId_embedding/ExpandDims:output:00userId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:���������2)
'userId_embedding/to_sparse_input/values�
,userId_embedding/to_sparse_input/dense_shapeShape$userId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2.
,userId_embedding/to_sparse_input/dense_shape�
userId_embedding/valuesCast0userId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:���������2
userId_embedding/values�
5userId_embedding/userId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 27
5userId_embedding/userId_embedding_weights/Slice/begin�
4userId_embedding/userId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:26
4userId_embedding/userId_embedding_weights/Slice/size�
/userId_embedding/userId_embedding_weights/SliceSlice5userId_embedding/to_sparse_input/dense_shape:output:0>userId_embedding/userId_embedding_weights/Slice/begin:output:0=userId_embedding/userId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:21
/userId_embedding/userId_embedding_weights/Slice�
/userId_embedding/userId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/userId_embedding/userId_embedding_weights/Const�
.userId_embedding/userId_embedding_weights/ProdProd8userId_embedding/userId_embedding_weights/Slice:output:08userId_embedding/userId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: 20
.userId_embedding/userId_embedding_weights/Prod�
:userId_embedding/userId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :2<
:userId_embedding/userId_embedding_weights/GatherV2/indices�
7userId_embedding/userId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7userId_embedding/userId_embedding_weights/GatherV2/axis�
2userId_embedding/userId_embedding_weights/GatherV2GatherV25userId_embedding/to_sparse_input/dense_shape:output:0CuserId_embedding/userId_embedding_weights/GatherV2/indices:output:0@userId_embedding/userId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 24
2userId_embedding/userId_embedding_weights/GatherV2�
0userId_embedding/userId_embedding_weights/Cast/xPack7userId_embedding/userId_embedding_weights/Prod:output:0;userId_embedding/userId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:22
0userId_embedding/userId_embedding_weights/Cast/x�
7userId_embedding/userId_embedding_weights/SparseReshapeSparseReshape0userId_embedding/to_sparse_input/indices:index:05userId_embedding/to_sparse_input/dense_shape:output:09userId_embedding/userId_embedding_weights/Cast/x:output:0*-
_output_shapes
:���������:29
7userId_embedding/userId_embedding_weights/SparseReshape�
@userId_embedding/userId_embedding_weights/SparseReshape/IdentityIdentityuserId_embedding/values:y:0*
T0	*#
_output_shapes
:���������2B
@userId_embedding/userId_embedding_weights/SparseReshape/Identity�
8userId_embedding/userId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2:
8userId_embedding/userId_embedding_weights/GreaterEqual/y�
6userId_embedding/userId_embedding_weights/GreaterEqualGreaterEqualIuserId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0AuserId_embedding/userId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:���������28
6userId_embedding/userId_embedding_weights/GreaterEqual�
/userId_embedding/userId_embedding_weights/WhereWhere:userId_embedding/userId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:���������21
/userId_embedding/userId_embedding_weights/Where�
7userId_embedding/userId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������29
7userId_embedding/userId_embedding_weights/Reshape/shape�
1userId_embedding/userId_embedding_weights/ReshapeReshape7userId_embedding/userId_embedding_weights/Where:index:0@userId_embedding/userId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:���������23
1userId_embedding/userId_embedding_weights/Reshape�
9userId_embedding/userId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9userId_embedding/userId_embedding_weights/GatherV2_1/axis�
4userId_embedding/userId_embedding_weights/GatherV2_1GatherV2HuserId_embedding/userId_embedding_weights/SparseReshape:output_indices:0:userId_embedding/userId_embedding_weights/Reshape:output:0BuserId_embedding/userId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:���������26
4userId_embedding/userId_embedding_weights/GatherV2_1�
9userId_embedding/userId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9userId_embedding/userId_embedding_weights/GatherV2_2/axis�
4userId_embedding/userId_embedding_weights/GatherV2_2GatherV2IuserId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0:userId_embedding/userId_embedding_weights/Reshape:output:0BuserId_embedding/userId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:���������26
4userId_embedding/userId_embedding_weights/GatherV2_2�
2userId_embedding/userId_embedding_weights/IdentityIdentityFuserId_embedding/userId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:24
2userId_embedding/userId_embedding_weights/Identity�
CuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 2E
CuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/Const�
QuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows=userId_embedding/userId_embedding_weights/GatherV2_1:output:0=userId_embedding/userId_embedding_weights/GatherV2_2:output:0;userId_embedding/userId_embedding_weights/Identity:output:0LuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:���������:���������:���������:���������2S
QuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows�
UuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2W
UuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack�
WuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2Y
WuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1�
WuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2Y
WuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2�
OuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicebuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0^userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0`userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0`userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2Q
OuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice�
HuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/UniqueUniqueauserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:���������:���������2J
HuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique�
RuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherYuserid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_213264LuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*l
_classb
`^loc:@userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/213264*'
_output_shapes
:���������
*
dtype02T
RuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup�
[userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity[userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*l
_classb
`^loc:@userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/213264*'
_output_shapes
:���������
2]
[userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity�
]userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1IdentityduserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������
2_
]userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1�
AuserId_embedding/userId_embedding_weights/embedding_lookup_sparseSparseSegmentMeanfuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0NuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:idx:0XuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:���������
2C
AuserId_embedding/userId_embedding_weights/embedding_lookup_sparse�
9userId_embedding/userId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2;
9userId_embedding/userId_embedding_weights/Reshape_1/shape�
3userId_embedding/userId_embedding_weights/Reshape_1ReshapeguserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0BuserId_embedding/userId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:���������25
3userId_embedding/userId_embedding_weights/Reshape_1�
/userId_embedding/userId_embedding_weights/ShapeShapeJuserId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:21
/userId_embedding/userId_embedding_weights/Shape�
=userId_embedding/userId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
=userId_embedding/userId_embedding_weights/strided_slice/stack�
?userId_embedding/userId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?userId_embedding/userId_embedding_weights/strided_slice/stack_1�
?userId_embedding/userId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?userId_embedding/userId_embedding_weights/strided_slice/stack_2�
7userId_embedding/userId_embedding_weights/strided_sliceStridedSlice8userId_embedding/userId_embedding_weights/Shape:output:0FuserId_embedding/userId_embedding_weights/strided_slice/stack:output:0HuserId_embedding/userId_embedding_weights/strided_slice/stack_1:output:0HuserId_embedding/userId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7userId_embedding/userId_embedding_weights/strided_slice�
1userId_embedding/userId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :23
1userId_embedding/userId_embedding_weights/stack/0�
/userId_embedding/userId_embedding_weights/stackPack:userId_embedding/userId_embedding_weights/stack/0:output:0@userId_embedding/userId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:21
/userId_embedding/userId_embedding_weights/stack�
.userId_embedding/userId_embedding_weights/TileTile<userId_embedding/userId_embedding_weights/Reshape_1:output:08userId_embedding/userId_embedding_weights/stack:output:0*
T0
*0
_output_shapes
:������������������20
.userId_embedding/userId_embedding_weights/Tile�
4userId_embedding/userId_embedding_weights/zeros_like	ZerosLikeJuserId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
26
4userId_embedding/userId_embedding_weights/zeros_like�
)userId_embedding/userId_embedding_weightsSelect7userId_embedding/userId_embedding_weights/Tile:output:08userId_embedding/userId_embedding_weights/zeros_like:y:0JuserId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
2+
)userId_embedding/userId_embedding_weights�
0userId_embedding/userId_embedding_weights/Cast_1Cast5userId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:22
0userId_embedding/userId_embedding_weights/Cast_1�
7userId_embedding/userId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 29
7userId_embedding/userId_embedding_weights/Slice_1/begin�
6userId_embedding/userId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:28
6userId_embedding/userId_embedding_weights/Slice_1/size�
1userId_embedding/userId_embedding_weights/Slice_1Slice4userId_embedding/userId_embedding_weights/Cast_1:y:0@userId_embedding/userId_embedding_weights/Slice_1/begin:output:0?userId_embedding/userId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:23
1userId_embedding/userId_embedding_weights/Slice_1�
1userId_embedding/userId_embedding_weights/Shape_1Shape2userId_embedding/userId_embedding_weights:output:0*
T0*
_output_shapes
:23
1userId_embedding/userId_embedding_weights/Shape_1�
7userId_embedding/userId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:29
7userId_embedding/userId_embedding_weights/Slice_2/begin�
6userId_embedding/userId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
���������28
6userId_embedding/userId_embedding_weights/Slice_2/size�
1userId_embedding/userId_embedding_weights/Slice_2Slice:userId_embedding/userId_embedding_weights/Shape_1:output:0@userId_embedding/userId_embedding_weights/Slice_2/begin:output:0?userId_embedding/userId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:23
1userId_embedding/userId_embedding_weights/Slice_2�
5userId_embedding/userId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5userId_embedding/userId_embedding_weights/concat/axis�
0userId_embedding/userId_embedding_weights/concatConcatV2:userId_embedding/userId_embedding_weights/Slice_1:output:0:userId_embedding/userId_embedding_weights/Slice_2:output:0>userId_embedding/userId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0userId_embedding/userId_embedding_weights/concat�
3userId_embedding/userId_embedding_weights/Reshape_2Reshape2userId_embedding/userId_embedding_weights:output:09userId_embedding/userId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:���������
25
3userId_embedding/userId_embedding_weights/Reshape_2�
userId_embedding/ShapeShape<userId_embedding/userId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:2
userId_embedding/Shape�
$userId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$userId_embedding/strided_slice/stack�
&userId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&userId_embedding/strided_slice/stack_1�
&userId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&userId_embedding/strided_slice/stack_2�
userId_embedding/strided_sliceStridedSliceuserId_embedding/Shape:output:0-userId_embedding/strided_slice/stack:output:0/userId_embedding/strided_slice/stack_1:output:0/userId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
userId_embedding/strided_slice�
 userId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2"
 userId_embedding/Reshape/shape/1�
userId_embedding/Reshape/shapePack'userId_embedding/strided_slice:output:0)userId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
userId_embedding/Reshape/shape�
userId_embedding/ReshapeReshape<userId_embedding/userId_embedding_weights/Reshape_2:output:0'userId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:���������
2
userId_embedding/Reshapeq
concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat/concat_dim
concat/concatIdentity!userId_embedding/Reshape:output:0*
T0*'
_output_shapes
:���������
2
concat/concatj
IdentityIdentityconcat/concat:output:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*5
_input_shapes$
":���������:���������::U Q
#
_output_shapes
:���������
*
_user_specified_namefeatures/movieId:TP
#
_output_shapes
:���������
)
_user_specified_namefeatures/userId
�
�
1__inference_dense_features_1_layer_call_fn_213405
features_movieid
features_userid
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallfeatures_movieidfeatures_useridunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_dense_features_1_layer_call_and_return_conditional_losses_2123362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*5
_input_shapes$
":���������:���������:22
StatefulPartitionedCallStatefulPartitionedCall:U Q
#
_output_shapes
:���������
*
_user_specified_namefeatures/movieId:TP
#
_output_shapes
:���������
)
_user_specified_namefeatures/userId
��
�
"__inference__traced_restore_213758
file_prefix5
1assignvariableop_dense_features_embedding_weights9
5assignvariableop_1_dense_features_1_embedding_weights#
assignvariableop_2_dense_kernel!
assignvariableop_3_dense_bias%
!assignvariableop_4_dense_1_kernel#
assignvariableop_5_dense_1_bias%
!assignvariableop_6_dense_2_kernel#
assignvariableop_7_dense_2_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_1&
"assignvariableop_17_true_positives&
"assignvariableop_18_true_negatives'
#assignvariableop_19_false_positives'
#assignvariableop_20_false_negatives(
$assignvariableop_21_true_positives_1(
$assignvariableop_22_true_negatives_1)
%assignvariableop_23_false_positives_1)
%assignvariableop_24_false_negatives_1?
;assignvariableop_25_adam_dense_features_embedding_weights_mA
=assignvariableop_26_adam_dense_features_1_embedding_weights_m+
'assignvariableop_27_adam_dense_kernel_m)
%assignvariableop_28_adam_dense_bias_m-
)assignvariableop_29_adam_dense_1_kernel_m+
'assignvariableop_30_adam_dense_1_bias_m-
)assignvariableop_31_adam_dense_2_kernel_m+
'assignvariableop_32_adam_dense_2_bias_m?
;assignvariableop_33_adam_dense_features_embedding_weights_vA
=assignvariableop_34_adam_dense_features_1_embedding_weights_v+
'assignvariableop_35_adam_dense_kernel_v)
%assignvariableop_36_adam_dense_bias_v-
)assignvariableop_37_adam_dense_1_kernel_v+
'assignvariableop_38_adam_dense_1_bias_v-
)assignvariableop_39_adam_dense_2_kernel_v+
'assignvariableop_40_adam_dense_2_bias_v
identity_42��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*�
value�B�*BTlayer_with_weights-0/movieId_embedding.Sembedding_weights/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/userId_embedding.Sembedding_weights/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-0/movieId_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBolayer_with_weights-1/userId_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-0/movieId_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBolayer_with_weights-1/userId_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::*8
dtypes.
,2*	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp1assignvariableop_dense_features_embedding_weightsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp5assignvariableop_1_dense_features_1_embedding_weightsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp"assignvariableop_17_true_positivesIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp"assignvariableop_18_true_negativesIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp#assignvariableop_19_false_positivesIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp#assignvariableop_20_false_negativesIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp$assignvariableop_21_true_positives_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp$assignvariableop_22_true_negatives_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp%assignvariableop_23_false_positives_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp%assignvariableop_24_false_negatives_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp;assignvariableop_25_adam_dense_features_embedding_weights_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp=assignvariableop_26_adam_dense_features_1_embedding_weights_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dense_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_dense_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp;assignvariableop_33_adam_dense_features_embedding_weights_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp=assignvariableop_34_adam_dense_features_1_embedding_weights_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adam_dense_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp%assignvariableop_36_adam_dense_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_1_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_1_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_2_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_dense_2_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_409
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_41Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_41�
Identity_42IdentityIdentity_41:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_42"#
identity_42Identity_42:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
s
G__inference_concatenate_layer_call_and_return_conditional_losses_213412
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������
:���������
:Q M
'
_output_shapes
:���������

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������

"
_user_specified_name
inputs/1
�
{
&__inference_dense_layer_call_fn_213438

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2123842
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_212613
movieid

userid
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallmovieiduseridunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_2119702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:���������:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	movieId:KG
#
_output_shapes
:���������
 
_user_specified_nameuserId
�
�
C__inference_dense_2_layer_call_and_return_conditional_losses_213469

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
:::O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
��
�
J__inference_dense_features_layer_call_and_return_conditional_losses_213203
features_movieid
features_userid_
[movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_213163
identity��
 movieId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 movieId_embedding/ExpandDims/dim�
movieId_embedding/ExpandDims
ExpandDimsfeatures_movieid)movieId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������2
movieId_embedding/ExpandDims�
0movieId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
���������22
0movieId_embedding/to_sparse_input/ignore_value/x�
*movieId_embedding/to_sparse_input/NotEqualNotEqual%movieId_embedding/ExpandDims:output:09movieId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:���������2,
*movieId_embedding/to_sparse_input/NotEqual�
)movieId_embedding/to_sparse_input/indicesWhere.movieId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:���������2+
)movieId_embedding/to_sparse_input/indices�
(movieId_embedding/to_sparse_input/valuesGatherNd%movieId_embedding/ExpandDims:output:01movieId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:���������2*
(movieId_embedding/to_sparse_input/values�
-movieId_embedding/to_sparse_input/dense_shapeShape%movieId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2/
-movieId_embedding/to_sparse_input/dense_shape�
movieId_embedding/valuesCast1movieId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:���������2
movieId_embedding/values�
7movieId_embedding/movieId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 29
7movieId_embedding/movieId_embedding_weights/Slice/begin�
6movieId_embedding/movieId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:28
6movieId_embedding/movieId_embedding_weights/Slice/size�
1movieId_embedding/movieId_embedding_weights/SliceSlice6movieId_embedding/to_sparse_input/dense_shape:output:0@movieId_embedding/movieId_embedding_weights/Slice/begin:output:0?movieId_embedding/movieId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:23
1movieId_embedding/movieId_embedding_weights/Slice�
1movieId_embedding/movieId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1movieId_embedding/movieId_embedding_weights/Const�
0movieId_embedding/movieId_embedding_weights/ProdProd:movieId_embedding/movieId_embedding_weights/Slice:output:0:movieId_embedding/movieId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: 22
0movieId_embedding/movieId_embedding_weights/Prod�
<movieId_embedding/movieId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :2>
<movieId_embedding/movieId_embedding_weights/GatherV2/indices�
9movieId_embedding/movieId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9movieId_embedding/movieId_embedding_weights/GatherV2/axis�
4movieId_embedding/movieId_embedding_weights/GatherV2GatherV26movieId_embedding/to_sparse_input/dense_shape:output:0EmovieId_embedding/movieId_embedding_weights/GatherV2/indices:output:0BmovieId_embedding/movieId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 26
4movieId_embedding/movieId_embedding_weights/GatherV2�
2movieId_embedding/movieId_embedding_weights/Cast/xPack9movieId_embedding/movieId_embedding_weights/Prod:output:0=movieId_embedding/movieId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:24
2movieId_embedding/movieId_embedding_weights/Cast/x�
9movieId_embedding/movieId_embedding_weights/SparseReshapeSparseReshape1movieId_embedding/to_sparse_input/indices:index:06movieId_embedding/to_sparse_input/dense_shape:output:0;movieId_embedding/movieId_embedding_weights/Cast/x:output:0*-
_output_shapes
:���������:2;
9movieId_embedding/movieId_embedding_weights/SparseReshape�
BmovieId_embedding/movieId_embedding_weights/SparseReshape/IdentityIdentitymovieId_embedding/values:y:0*
T0	*#
_output_shapes
:���������2D
BmovieId_embedding/movieId_embedding_weights/SparseReshape/Identity�
:movieId_embedding/movieId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2<
:movieId_embedding/movieId_embedding_weights/GreaterEqual/y�
8movieId_embedding/movieId_embedding_weights/GreaterEqualGreaterEqualKmovieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0CmovieId_embedding/movieId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:���������2:
8movieId_embedding/movieId_embedding_weights/GreaterEqual�
1movieId_embedding/movieId_embedding_weights/WhereWhere<movieId_embedding/movieId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:���������23
1movieId_embedding/movieId_embedding_weights/Where�
9movieId_embedding/movieId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������2;
9movieId_embedding/movieId_embedding_weights/Reshape/shape�
3movieId_embedding/movieId_embedding_weights/ReshapeReshape9movieId_embedding/movieId_embedding_weights/Where:index:0BmovieId_embedding/movieId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:���������25
3movieId_embedding/movieId_embedding_weights/Reshape�
;movieId_embedding/movieId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;movieId_embedding/movieId_embedding_weights/GatherV2_1/axis�
6movieId_embedding/movieId_embedding_weights/GatherV2_1GatherV2JmovieId_embedding/movieId_embedding_weights/SparseReshape:output_indices:0<movieId_embedding/movieId_embedding_weights/Reshape:output:0DmovieId_embedding/movieId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:���������28
6movieId_embedding/movieId_embedding_weights/GatherV2_1�
;movieId_embedding/movieId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;movieId_embedding/movieId_embedding_weights/GatherV2_2/axis�
6movieId_embedding/movieId_embedding_weights/GatherV2_2GatherV2KmovieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0<movieId_embedding/movieId_embedding_weights/Reshape:output:0DmovieId_embedding/movieId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:���������28
6movieId_embedding/movieId_embedding_weights/GatherV2_2�
4movieId_embedding/movieId_embedding_weights/IdentityIdentityHmovieId_embedding/movieId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:26
4movieId_embedding/movieId_embedding_weights/Identity�
EmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 2G
EmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/Const�
SmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows?movieId_embedding/movieId_embedding_weights/GatherV2_1:output:0?movieId_embedding/movieId_embedding_weights/GatherV2_2:output:0=movieId_embedding/movieId_embedding_weights/Identity:output:0NmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:���������:���������:���������:���������2U
SmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows�
WmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2Y
WmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack�
YmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2[
YmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1�
YmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2[
YmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2�
QmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicedmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0`movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0bmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0bmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2S
QmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice�
JmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/UniqueUniquecmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:���������:���������2L
JmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique�
TmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGather[movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_213163NmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*n
_classd
b`loc:@movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/213163*'
_output_shapes
:���������
*
dtype02V
TmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup�
]movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity]movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*n
_classd
b`loc:@movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/213163*'
_output_shapes
:���������
2_
]movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity�
_movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1IdentityfmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������
2a
_movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1�
CmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparseSparseSegmentMeanhmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0PmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:idx:0ZmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:���������
2E
CmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse�
;movieId_embedding/movieId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2=
;movieId_embedding/movieId_embedding_weights/Reshape_1/shape�
5movieId_embedding/movieId_embedding_weights/Reshape_1ReshapeimovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0DmovieId_embedding/movieId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:���������27
5movieId_embedding/movieId_embedding_weights/Reshape_1�
1movieId_embedding/movieId_embedding_weights/ShapeShapeLmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:23
1movieId_embedding/movieId_embedding_weights/Shape�
?movieId_embedding/movieId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2A
?movieId_embedding/movieId_embedding_weights/strided_slice/stack�
AmovieId_embedding/movieId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
AmovieId_embedding/movieId_embedding_weights/strided_slice/stack_1�
AmovieId_embedding/movieId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
AmovieId_embedding/movieId_embedding_weights/strided_slice/stack_2�
9movieId_embedding/movieId_embedding_weights/strided_sliceStridedSlice:movieId_embedding/movieId_embedding_weights/Shape:output:0HmovieId_embedding/movieId_embedding_weights/strided_slice/stack:output:0JmovieId_embedding/movieId_embedding_weights/strided_slice/stack_1:output:0JmovieId_embedding/movieId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9movieId_embedding/movieId_embedding_weights/strided_slice�
3movieId_embedding/movieId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :25
3movieId_embedding/movieId_embedding_weights/stack/0�
1movieId_embedding/movieId_embedding_weights/stackPack<movieId_embedding/movieId_embedding_weights/stack/0:output:0BmovieId_embedding/movieId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:23
1movieId_embedding/movieId_embedding_weights/stack�
0movieId_embedding/movieId_embedding_weights/TileTile>movieId_embedding/movieId_embedding_weights/Reshape_1:output:0:movieId_embedding/movieId_embedding_weights/stack:output:0*
T0
*0
_output_shapes
:������������������22
0movieId_embedding/movieId_embedding_weights/Tile�
6movieId_embedding/movieId_embedding_weights/zeros_like	ZerosLikeLmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
28
6movieId_embedding/movieId_embedding_weights/zeros_like�
+movieId_embedding/movieId_embedding_weightsSelect9movieId_embedding/movieId_embedding_weights/Tile:output:0:movieId_embedding/movieId_embedding_weights/zeros_like:y:0LmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
2-
+movieId_embedding/movieId_embedding_weights�
2movieId_embedding/movieId_embedding_weights/Cast_1Cast6movieId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:24
2movieId_embedding/movieId_embedding_weights/Cast_1�
9movieId_embedding/movieId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 2;
9movieId_embedding/movieId_embedding_weights/Slice_1/begin�
8movieId_embedding/movieId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:2:
8movieId_embedding/movieId_embedding_weights/Slice_1/size�
3movieId_embedding/movieId_embedding_weights/Slice_1Slice6movieId_embedding/movieId_embedding_weights/Cast_1:y:0BmovieId_embedding/movieId_embedding_weights/Slice_1/begin:output:0AmovieId_embedding/movieId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:25
3movieId_embedding/movieId_embedding_weights/Slice_1�
3movieId_embedding/movieId_embedding_weights/Shape_1Shape4movieId_embedding/movieId_embedding_weights:output:0*
T0*
_output_shapes
:25
3movieId_embedding/movieId_embedding_weights/Shape_1�
9movieId_embedding/movieId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:2;
9movieId_embedding/movieId_embedding_weights/Slice_2/begin�
8movieId_embedding/movieId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
���������2:
8movieId_embedding/movieId_embedding_weights/Slice_2/size�
3movieId_embedding/movieId_embedding_weights/Slice_2Slice<movieId_embedding/movieId_embedding_weights/Shape_1:output:0BmovieId_embedding/movieId_embedding_weights/Slice_2/begin:output:0AmovieId_embedding/movieId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:25
3movieId_embedding/movieId_embedding_weights/Slice_2�
7movieId_embedding/movieId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7movieId_embedding/movieId_embedding_weights/concat/axis�
2movieId_embedding/movieId_embedding_weights/concatConcatV2<movieId_embedding/movieId_embedding_weights/Slice_1:output:0<movieId_embedding/movieId_embedding_weights/Slice_2:output:0@movieId_embedding/movieId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2movieId_embedding/movieId_embedding_weights/concat�
5movieId_embedding/movieId_embedding_weights/Reshape_2Reshape4movieId_embedding/movieId_embedding_weights:output:0;movieId_embedding/movieId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:���������
27
5movieId_embedding/movieId_embedding_weights/Reshape_2�
movieId_embedding/ShapeShape>movieId_embedding/movieId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:2
movieId_embedding/Shape�
%movieId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%movieId_embedding/strided_slice/stack�
'movieId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'movieId_embedding/strided_slice/stack_1�
'movieId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'movieId_embedding/strided_slice/stack_2�
movieId_embedding/strided_sliceStridedSlice movieId_embedding/Shape:output:0.movieId_embedding/strided_slice/stack:output:00movieId_embedding/strided_slice/stack_1:output:00movieId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
movieId_embedding/strided_slice�
!movieId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2#
!movieId_embedding/Reshape/shape/1�
movieId_embedding/Reshape/shapePack(movieId_embedding/strided_slice:output:0*movieId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2!
movieId_embedding/Reshape/shape�
movieId_embedding/ReshapeReshape>movieId_embedding/movieId_embedding_weights/Reshape_2:output:0(movieId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:���������
2
movieId_embedding/Reshapeq
concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat/concat_dim�
concat/concatIdentity"movieId_embedding/Reshape:output:0*
T0*'
_output_shapes
:���������
2
concat/concatj
IdentityIdentityconcat/concat:output:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*5
_input_shapes$
":���������:���������::U Q
#
_output_shapes
:���������
*
_user_specified_namefeatures/movieId:TP
#
_output_shapes
:���������
)
_user_specified_namefeatures/userId
�
�
H__inference_functional_1_layer_call_and_return_conditional_losses_212482
movieid

userid
dense_features_212459
dense_features_1_212462
dense_212466
dense_212468
dense_1_212471
dense_1_212473
dense_2_212476
dense_2_212478
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�&dense_features/StatefulPartitionedCall�(dense_features_1/StatefulPartitionedCall�
&dense_features/StatefulPartitionedCallStatefulPartitionedCallmovieiduseriddense_features_212459*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_2121452(
&dense_features/StatefulPartitionedCall�
(dense_features_1/StatefulPartitionedCallStatefulPartitionedCallmovieiduseriddense_features_1_212462*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_dense_features_1_layer_call_and_return_conditional_losses_2123362*
(dense_features_1/StatefulPartitionedCall�
concatenate/PartitionedCallPartitionedCall/dense_features/StatefulPartitionedCall:output:01dense_features_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_2123642
concatenate/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_212466dense_212468*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2123842
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_212471dense_1_212473*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2124112!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_212476dense_2_212478*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2124382!
dense_2/StatefulPartitionedCall�
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall'^dense_features/StatefulPartitionedCall)^dense_features_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:���������:���������::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2P
&dense_features/StatefulPartitionedCall&dense_features/StatefulPartitionedCall2T
(dense_features_1/StatefulPartitionedCall(dense_features_1/StatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	movieId:KG
#
_output_shapes
:���������
 
_user_specified_nameuserId
�
�
H__inference_functional_1_layer_call_and_return_conditional_losses_212455
movieid

userid
dense_features_212163
dense_features_1_212354
dense_212395
dense_212397
dense_1_212422
dense_1_212424
dense_2_212449
dense_2_212451
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�&dense_features/StatefulPartitionedCall�(dense_features_1/StatefulPartitionedCall�
&dense_features/StatefulPartitionedCallStatefulPartitionedCallmovieiduseriddense_features_212163*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_2120602(
&dense_features/StatefulPartitionedCall�
(dense_features_1/StatefulPartitionedCallStatefulPartitionedCallmovieiduseriddense_features_1_212354*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_dense_features_1_layer_call_and_return_conditional_losses_2122512*
(dense_features_1/StatefulPartitionedCall�
concatenate/PartitionedCallPartitionedCall/dense_features/StatefulPartitionedCall:output:01dense_features_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_2123642
concatenate/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_212395dense_212397*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2123842
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_212422dense_1_212424*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2124112!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_212449dense_2_212451*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2124382!
dense_2/StatefulPartitionedCall�
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall'^dense_features/StatefulPartitionedCall)^dense_features_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:���������:���������::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2P
&dense_features/StatefulPartitionedCall&dense_features/StatefulPartitionedCall2T
(dense_features_1/StatefulPartitionedCall(dense_features_1/StatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	movieId:KG
#
_output_shapes
:���������
 
_user_specified_nameuserId
��
�
J__inference_dense_features_layer_call_and_return_conditional_losses_213118
features_movieid
features_userid_
[movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_213078
identity��
 movieId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 movieId_embedding/ExpandDims/dim�
movieId_embedding/ExpandDims
ExpandDimsfeatures_movieid)movieId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������2
movieId_embedding/ExpandDims�
0movieId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
���������22
0movieId_embedding/to_sparse_input/ignore_value/x�
*movieId_embedding/to_sparse_input/NotEqualNotEqual%movieId_embedding/ExpandDims:output:09movieId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:���������2,
*movieId_embedding/to_sparse_input/NotEqual�
)movieId_embedding/to_sparse_input/indicesWhere.movieId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:���������2+
)movieId_embedding/to_sparse_input/indices�
(movieId_embedding/to_sparse_input/valuesGatherNd%movieId_embedding/ExpandDims:output:01movieId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:���������2*
(movieId_embedding/to_sparse_input/values�
-movieId_embedding/to_sparse_input/dense_shapeShape%movieId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2/
-movieId_embedding/to_sparse_input/dense_shape�
movieId_embedding/valuesCast1movieId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:���������2
movieId_embedding/values�
7movieId_embedding/movieId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 29
7movieId_embedding/movieId_embedding_weights/Slice/begin�
6movieId_embedding/movieId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:28
6movieId_embedding/movieId_embedding_weights/Slice/size�
1movieId_embedding/movieId_embedding_weights/SliceSlice6movieId_embedding/to_sparse_input/dense_shape:output:0@movieId_embedding/movieId_embedding_weights/Slice/begin:output:0?movieId_embedding/movieId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:23
1movieId_embedding/movieId_embedding_weights/Slice�
1movieId_embedding/movieId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1movieId_embedding/movieId_embedding_weights/Const�
0movieId_embedding/movieId_embedding_weights/ProdProd:movieId_embedding/movieId_embedding_weights/Slice:output:0:movieId_embedding/movieId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: 22
0movieId_embedding/movieId_embedding_weights/Prod�
<movieId_embedding/movieId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :2>
<movieId_embedding/movieId_embedding_weights/GatherV2/indices�
9movieId_embedding/movieId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9movieId_embedding/movieId_embedding_weights/GatherV2/axis�
4movieId_embedding/movieId_embedding_weights/GatherV2GatherV26movieId_embedding/to_sparse_input/dense_shape:output:0EmovieId_embedding/movieId_embedding_weights/GatherV2/indices:output:0BmovieId_embedding/movieId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 26
4movieId_embedding/movieId_embedding_weights/GatherV2�
2movieId_embedding/movieId_embedding_weights/Cast/xPack9movieId_embedding/movieId_embedding_weights/Prod:output:0=movieId_embedding/movieId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:24
2movieId_embedding/movieId_embedding_weights/Cast/x�
9movieId_embedding/movieId_embedding_weights/SparseReshapeSparseReshape1movieId_embedding/to_sparse_input/indices:index:06movieId_embedding/to_sparse_input/dense_shape:output:0;movieId_embedding/movieId_embedding_weights/Cast/x:output:0*-
_output_shapes
:���������:2;
9movieId_embedding/movieId_embedding_weights/SparseReshape�
BmovieId_embedding/movieId_embedding_weights/SparseReshape/IdentityIdentitymovieId_embedding/values:y:0*
T0	*#
_output_shapes
:���������2D
BmovieId_embedding/movieId_embedding_weights/SparseReshape/Identity�
:movieId_embedding/movieId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2<
:movieId_embedding/movieId_embedding_weights/GreaterEqual/y�
8movieId_embedding/movieId_embedding_weights/GreaterEqualGreaterEqualKmovieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0CmovieId_embedding/movieId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:���������2:
8movieId_embedding/movieId_embedding_weights/GreaterEqual�
1movieId_embedding/movieId_embedding_weights/WhereWhere<movieId_embedding/movieId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:���������23
1movieId_embedding/movieId_embedding_weights/Where�
9movieId_embedding/movieId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������2;
9movieId_embedding/movieId_embedding_weights/Reshape/shape�
3movieId_embedding/movieId_embedding_weights/ReshapeReshape9movieId_embedding/movieId_embedding_weights/Where:index:0BmovieId_embedding/movieId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:���������25
3movieId_embedding/movieId_embedding_weights/Reshape�
;movieId_embedding/movieId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;movieId_embedding/movieId_embedding_weights/GatherV2_1/axis�
6movieId_embedding/movieId_embedding_weights/GatherV2_1GatherV2JmovieId_embedding/movieId_embedding_weights/SparseReshape:output_indices:0<movieId_embedding/movieId_embedding_weights/Reshape:output:0DmovieId_embedding/movieId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:���������28
6movieId_embedding/movieId_embedding_weights/GatherV2_1�
;movieId_embedding/movieId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;movieId_embedding/movieId_embedding_weights/GatherV2_2/axis�
6movieId_embedding/movieId_embedding_weights/GatherV2_2GatherV2KmovieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0<movieId_embedding/movieId_embedding_weights/Reshape:output:0DmovieId_embedding/movieId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:���������28
6movieId_embedding/movieId_embedding_weights/GatherV2_2�
4movieId_embedding/movieId_embedding_weights/IdentityIdentityHmovieId_embedding/movieId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:26
4movieId_embedding/movieId_embedding_weights/Identity�
EmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 2G
EmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/Const�
SmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows?movieId_embedding/movieId_embedding_weights/GatherV2_1:output:0?movieId_embedding/movieId_embedding_weights/GatherV2_2:output:0=movieId_embedding/movieId_embedding_weights/Identity:output:0NmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:���������:���������:���������:���������2U
SmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows�
WmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2Y
WmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack�
YmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2[
YmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1�
YmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2[
YmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2�
QmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicedmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0`movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0bmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0bmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2S
QmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice�
JmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/UniqueUniquecmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:���������:���������2L
JmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique�
TmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGather[movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_213078NmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*n
_classd
b`loc:@movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/213078*'
_output_shapes
:���������
*
dtype02V
TmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup�
]movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity]movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*n
_classd
b`loc:@movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/213078*'
_output_shapes
:���������
2_
]movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity�
_movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1IdentityfmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������
2a
_movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1�
CmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparseSparseSegmentMeanhmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0PmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:idx:0ZmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:���������
2E
CmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse�
;movieId_embedding/movieId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2=
;movieId_embedding/movieId_embedding_weights/Reshape_1/shape�
5movieId_embedding/movieId_embedding_weights/Reshape_1ReshapeimovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0DmovieId_embedding/movieId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:���������27
5movieId_embedding/movieId_embedding_weights/Reshape_1�
1movieId_embedding/movieId_embedding_weights/ShapeShapeLmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:23
1movieId_embedding/movieId_embedding_weights/Shape�
?movieId_embedding/movieId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2A
?movieId_embedding/movieId_embedding_weights/strided_slice/stack�
AmovieId_embedding/movieId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
AmovieId_embedding/movieId_embedding_weights/strided_slice/stack_1�
AmovieId_embedding/movieId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
AmovieId_embedding/movieId_embedding_weights/strided_slice/stack_2�
9movieId_embedding/movieId_embedding_weights/strided_sliceStridedSlice:movieId_embedding/movieId_embedding_weights/Shape:output:0HmovieId_embedding/movieId_embedding_weights/strided_slice/stack:output:0JmovieId_embedding/movieId_embedding_weights/strided_slice/stack_1:output:0JmovieId_embedding/movieId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9movieId_embedding/movieId_embedding_weights/strided_slice�
3movieId_embedding/movieId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :25
3movieId_embedding/movieId_embedding_weights/stack/0�
1movieId_embedding/movieId_embedding_weights/stackPack<movieId_embedding/movieId_embedding_weights/stack/0:output:0BmovieId_embedding/movieId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:23
1movieId_embedding/movieId_embedding_weights/stack�
0movieId_embedding/movieId_embedding_weights/TileTile>movieId_embedding/movieId_embedding_weights/Reshape_1:output:0:movieId_embedding/movieId_embedding_weights/stack:output:0*
T0
*0
_output_shapes
:������������������22
0movieId_embedding/movieId_embedding_weights/Tile�
6movieId_embedding/movieId_embedding_weights/zeros_like	ZerosLikeLmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
28
6movieId_embedding/movieId_embedding_weights/zeros_like�
+movieId_embedding/movieId_embedding_weightsSelect9movieId_embedding/movieId_embedding_weights/Tile:output:0:movieId_embedding/movieId_embedding_weights/zeros_like:y:0LmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
2-
+movieId_embedding/movieId_embedding_weights�
2movieId_embedding/movieId_embedding_weights/Cast_1Cast6movieId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:24
2movieId_embedding/movieId_embedding_weights/Cast_1�
9movieId_embedding/movieId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 2;
9movieId_embedding/movieId_embedding_weights/Slice_1/begin�
8movieId_embedding/movieId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:2:
8movieId_embedding/movieId_embedding_weights/Slice_1/size�
3movieId_embedding/movieId_embedding_weights/Slice_1Slice6movieId_embedding/movieId_embedding_weights/Cast_1:y:0BmovieId_embedding/movieId_embedding_weights/Slice_1/begin:output:0AmovieId_embedding/movieId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:25
3movieId_embedding/movieId_embedding_weights/Slice_1�
3movieId_embedding/movieId_embedding_weights/Shape_1Shape4movieId_embedding/movieId_embedding_weights:output:0*
T0*
_output_shapes
:25
3movieId_embedding/movieId_embedding_weights/Shape_1�
9movieId_embedding/movieId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:2;
9movieId_embedding/movieId_embedding_weights/Slice_2/begin�
8movieId_embedding/movieId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
���������2:
8movieId_embedding/movieId_embedding_weights/Slice_2/size�
3movieId_embedding/movieId_embedding_weights/Slice_2Slice<movieId_embedding/movieId_embedding_weights/Shape_1:output:0BmovieId_embedding/movieId_embedding_weights/Slice_2/begin:output:0AmovieId_embedding/movieId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:25
3movieId_embedding/movieId_embedding_weights/Slice_2�
7movieId_embedding/movieId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7movieId_embedding/movieId_embedding_weights/concat/axis�
2movieId_embedding/movieId_embedding_weights/concatConcatV2<movieId_embedding/movieId_embedding_weights/Slice_1:output:0<movieId_embedding/movieId_embedding_weights/Slice_2:output:0@movieId_embedding/movieId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2movieId_embedding/movieId_embedding_weights/concat�
5movieId_embedding/movieId_embedding_weights/Reshape_2Reshape4movieId_embedding/movieId_embedding_weights:output:0;movieId_embedding/movieId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:���������
27
5movieId_embedding/movieId_embedding_weights/Reshape_2�
movieId_embedding/ShapeShape>movieId_embedding/movieId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:2
movieId_embedding/Shape�
%movieId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%movieId_embedding/strided_slice/stack�
'movieId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'movieId_embedding/strided_slice/stack_1�
'movieId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'movieId_embedding/strided_slice/stack_2�
movieId_embedding/strided_sliceStridedSlice movieId_embedding/Shape:output:0.movieId_embedding/strided_slice/stack:output:00movieId_embedding/strided_slice/stack_1:output:00movieId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
movieId_embedding/strided_slice�
!movieId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2#
!movieId_embedding/Reshape/shape/1�
movieId_embedding/Reshape/shapePack(movieId_embedding/strided_slice:output:0*movieId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2!
movieId_embedding/Reshape/shape�
movieId_embedding/ReshapeReshape>movieId_embedding/movieId_embedding_weights/Reshape_2:output:0(movieId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:���������
2
movieId_embedding/Reshapeq
concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat/concat_dim�
concat/concatIdentity"movieId_embedding/Reshape:output:0*
T0*'
_output_shapes
:���������
2
concat/concatj
IdentityIdentityconcat/concat:output:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*5
_input_shapes$
":���������:���������::U Q
#
_output_shapes
:���������
*
_user_specified_namefeatures/movieId:TP
#
_output_shapes
:���������
)
_user_specified_namefeatures/userId
�
�
A__inference_dense_layer_call_and_return_conditional_losses_212384

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������
2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
ј
�
L__inference_dense_features_1_layer_call_and_return_conditional_losses_212336
features

features_1]
Yuserid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_212296
identity��
userId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
userId_embedding/ExpandDims/dim�
userId_embedding/ExpandDims
ExpandDims
features_1(userId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������2
userId_embedding/ExpandDims�
/userId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
���������21
/userId_embedding/to_sparse_input/ignore_value/x�
)userId_embedding/to_sparse_input/NotEqualNotEqual$userId_embedding/ExpandDims:output:08userId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:���������2+
)userId_embedding/to_sparse_input/NotEqual�
(userId_embedding/to_sparse_input/indicesWhere-userId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:���������2*
(userId_embedding/to_sparse_input/indices�
'userId_embedding/to_sparse_input/valuesGatherNd$userId_embedding/ExpandDims:output:00userId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:���������2)
'userId_embedding/to_sparse_input/values�
,userId_embedding/to_sparse_input/dense_shapeShape$userId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2.
,userId_embedding/to_sparse_input/dense_shape�
userId_embedding/valuesCast0userId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:���������2
userId_embedding/values�
5userId_embedding/userId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 27
5userId_embedding/userId_embedding_weights/Slice/begin�
4userId_embedding/userId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:26
4userId_embedding/userId_embedding_weights/Slice/size�
/userId_embedding/userId_embedding_weights/SliceSlice5userId_embedding/to_sparse_input/dense_shape:output:0>userId_embedding/userId_embedding_weights/Slice/begin:output:0=userId_embedding/userId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:21
/userId_embedding/userId_embedding_weights/Slice�
/userId_embedding/userId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/userId_embedding/userId_embedding_weights/Const�
.userId_embedding/userId_embedding_weights/ProdProd8userId_embedding/userId_embedding_weights/Slice:output:08userId_embedding/userId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: 20
.userId_embedding/userId_embedding_weights/Prod�
:userId_embedding/userId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :2<
:userId_embedding/userId_embedding_weights/GatherV2/indices�
7userId_embedding/userId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7userId_embedding/userId_embedding_weights/GatherV2/axis�
2userId_embedding/userId_embedding_weights/GatherV2GatherV25userId_embedding/to_sparse_input/dense_shape:output:0CuserId_embedding/userId_embedding_weights/GatherV2/indices:output:0@userId_embedding/userId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 24
2userId_embedding/userId_embedding_weights/GatherV2�
0userId_embedding/userId_embedding_weights/Cast/xPack7userId_embedding/userId_embedding_weights/Prod:output:0;userId_embedding/userId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:22
0userId_embedding/userId_embedding_weights/Cast/x�
7userId_embedding/userId_embedding_weights/SparseReshapeSparseReshape0userId_embedding/to_sparse_input/indices:index:05userId_embedding/to_sparse_input/dense_shape:output:09userId_embedding/userId_embedding_weights/Cast/x:output:0*-
_output_shapes
:���������:29
7userId_embedding/userId_embedding_weights/SparseReshape�
@userId_embedding/userId_embedding_weights/SparseReshape/IdentityIdentityuserId_embedding/values:y:0*
T0	*#
_output_shapes
:���������2B
@userId_embedding/userId_embedding_weights/SparseReshape/Identity�
8userId_embedding/userId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2:
8userId_embedding/userId_embedding_weights/GreaterEqual/y�
6userId_embedding/userId_embedding_weights/GreaterEqualGreaterEqualIuserId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0AuserId_embedding/userId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:���������28
6userId_embedding/userId_embedding_weights/GreaterEqual�
/userId_embedding/userId_embedding_weights/WhereWhere:userId_embedding/userId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:���������21
/userId_embedding/userId_embedding_weights/Where�
7userId_embedding/userId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������29
7userId_embedding/userId_embedding_weights/Reshape/shape�
1userId_embedding/userId_embedding_weights/ReshapeReshape7userId_embedding/userId_embedding_weights/Where:index:0@userId_embedding/userId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:���������23
1userId_embedding/userId_embedding_weights/Reshape�
9userId_embedding/userId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9userId_embedding/userId_embedding_weights/GatherV2_1/axis�
4userId_embedding/userId_embedding_weights/GatherV2_1GatherV2HuserId_embedding/userId_embedding_weights/SparseReshape:output_indices:0:userId_embedding/userId_embedding_weights/Reshape:output:0BuserId_embedding/userId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:���������26
4userId_embedding/userId_embedding_weights/GatherV2_1�
9userId_embedding/userId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9userId_embedding/userId_embedding_weights/GatherV2_2/axis�
4userId_embedding/userId_embedding_weights/GatherV2_2GatherV2IuserId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0:userId_embedding/userId_embedding_weights/Reshape:output:0BuserId_embedding/userId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:���������26
4userId_embedding/userId_embedding_weights/GatherV2_2�
2userId_embedding/userId_embedding_weights/IdentityIdentityFuserId_embedding/userId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:24
2userId_embedding/userId_embedding_weights/Identity�
CuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 2E
CuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/Const�
QuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows=userId_embedding/userId_embedding_weights/GatherV2_1:output:0=userId_embedding/userId_embedding_weights/GatherV2_2:output:0;userId_embedding/userId_embedding_weights/Identity:output:0LuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:���������:���������:���������:���������2S
QuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows�
UuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2W
UuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack�
WuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2Y
WuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1�
WuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2Y
WuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2�
OuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicebuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0^userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0`userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0`userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2Q
OuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice�
HuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/UniqueUniqueauserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:���������:���������2J
HuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique�
RuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherYuserid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_212296LuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*l
_classb
`^loc:@userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/212296*'
_output_shapes
:���������
*
dtype02T
RuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup�
[userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity[userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*l
_classb
`^loc:@userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/212296*'
_output_shapes
:���������
2]
[userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity�
]userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1IdentityduserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������
2_
]userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1�
AuserId_embedding/userId_embedding_weights/embedding_lookup_sparseSparseSegmentMeanfuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0NuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:idx:0XuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:���������
2C
AuserId_embedding/userId_embedding_weights/embedding_lookup_sparse�
9userId_embedding/userId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2;
9userId_embedding/userId_embedding_weights/Reshape_1/shape�
3userId_embedding/userId_embedding_weights/Reshape_1ReshapeguserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0BuserId_embedding/userId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:���������25
3userId_embedding/userId_embedding_weights/Reshape_1�
/userId_embedding/userId_embedding_weights/ShapeShapeJuserId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:21
/userId_embedding/userId_embedding_weights/Shape�
=userId_embedding/userId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
=userId_embedding/userId_embedding_weights/strided_slice/stack�
?userId_embedding/userId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?userId_embedding/userId_embedding_weights/strided_slice/stack_1�
?userId_embedding/userId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?userId_embedding/userId_embedding_weights/strided_slice/stack_2�
7userId_embedding/userId_embedding_weights/strided_sliceStridedSlice8userId_embedding/userId_embedding_weights/Shape:output:0FuserId_embedding/userId_embedding_weights/strided_slice/stack:output:0HuserId_embedding/userId_embedding_weights/strided_slice/stack_1:output:0HuserId_embedding/userId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7userId_embedding/userId_embedding_weights/strided_slice�
1userId_embedding/userId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :23
1userId_embedding/userId_embedding_weights/stack/0�
/userId_embedding/userId_embedding_weights/stackPack:userId_embedding/userId_embedding_weights/stack/0:output:0@userId_embedding/userId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:21
/userId_embedding/userId_embedding_weights/stack�
.userId_embedding/userId_embedding_weights/TileTile<userId_embedding/userId_embedding_weights/Reshape_1:output:08userId_embedding/userId_embedding_weights/stack:output:0*
T0
*0
_output_shapes
:������������������20
.userId_embedding/userId_embedding_weights/Tile�
4userId_embedding/userId_embedding_weights/zeros_like	ZerosLikeJuserId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
26
4userId_embedding/userId_embedding_weights/zeros_like�
)userId_embedding/userId_embedding_weightsSelect7userId_embedding/userId_embedding_weights/Tile:output:08userId_embedding/userId_embedding_weights/zeros_like:y:0JuserId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
2+
)userId_embedding/userId_embedding_weights�
0userId_embedding/userId_embedding_weights/Cast_1Cast5userId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:22
0userId_embedding/userId_embedding_weights/Cast_1�
7userId_embedding/userId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 29
7userId_embedding/userId_embedding_weights/Slice_1/begin�
6userId_embedding/userId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:28
6userId_embedding/userId_embedding_weights/Slice_1/size�
1userId_embedding/userId_embedding_weights/Slice_1Slice4userId_embedding/userId_embedding_weights/Cast_1:y:0@userId_embedding/userId_embedding_weights/Slice_1/begin:output:0?userId_embedding/userId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:23
1userId_embedding/userId_embedding_weights/Slice_1�
1userId_embedding/userId_embedding_weights/Shape_1Shape2userId_embedding/userId_embedding_weights:output:0*
T0*
_output_shapes
:23
1userId_embedding/userId_embedding_weights/Shape_1�
7userId_embedding/userId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:29
7userId_embedding/userId_embedding_weights/Slice_2/begin�
6userId_embedding/userId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
���������28
6userId_embedding/userId_embedding_weights/Slice_2/size�
1userId_embedding/userId_embedding_weights/Slice_2Slice:userId_embedding/userId_embedding_weights/Shape_1:output:0@userId_embedding/userId_embedding_weights/Slice_2/begin:output:0?userId_embedding/userId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:23
1userId_embedding/userId_embedding_weights/Slice_2�
5userId_embedding/userId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5userId_embedding/userId_embedding_weights/concat/axis�
0userId_embedding/userId_embedding_weights/concatConcatV2:userId_embedding/userId_embedding_weights/Slice_1:output:0:userId_embedding/userId_embedding_weights/Slice_2:output:0>userId_embedding/userId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0userId_embedding/userId_embedding_weights/concat�
3userId_embedding/userId_embedding_weights/Reshape_2Reshape2userId_embedding/userId_embedding_weights:output:09userId_embedding/userId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:���������
25
3userId_embedding/userId_embedding_weights/Reshape_2�
userId_embedding/ShapeShape<userId_embedding/userId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:2
userId_embedding/Shape�
$userId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$userId_embedding/strided_slice/stack�
&userId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&userId_embedding/strided_slice/stack_1�
&userId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&userId_embedding/strided_slice/stack_2�
userId_embedding/strided_sliceStridedSliceuserId_embedding/Shape:output:0-userId_embedding/strided_slice/stack:output:0/userId_embedding/strided_slice/stack_1:output:0/userId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
userId_embedding/strided_slice�
 userId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2"
 userId_embedding/Reshape/shape/1�
userId_embedding/Reshape/shapePack'userId_embedding/strided_slice:output:0)userId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
userId_embedding/Reshape/shape�
userId_embedding/ReshapeReshape<userId_embedding/userId_embedding_weights/Reshape_2:output:0'userId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:���������
2
userId_embedding/Reshapeq
concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat/concat_dim
concat/concatIdentity!userId_embedding/Reshape:output:0*
T0*'
_output_shapes
:���������
2
concat/concatj
IdentityIdentityconcat/concat:output:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*5
_input_shapes$
":���������:���������::M I
#
_output_shapes
:���������
"
_user_specified_name
features:MI
#
_output_shapes
:���������
"
_user_specified_name
features
ј
�
L__inference_dense_features_1_layer_call_and_return_conditional_losses_212251
features

features_1]
Yuserid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_212211
identity��
userId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
userId_embedding/ExpandDims/dim�
userId_embedding/ExpandDims
ExpandDims
features_1(userId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������2
userId_embedding/ExpandDims�
/userId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
���������21
/userId_embedding/to_sparse_input/ignore_value/x�
)userId_embedding/to_sparse_input/NotEqualNotEqual$userId_embedding/ExpandDims:output:08userId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:���������2+
)userId_embedding/to_sparse_input/NotEqual�
(userId_embedding/to_sparse_input/indicesWhere-userId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:���������2*
(userId_embedding/to_sparse_input/indices�
'userId_embedding/to_sparse_input/valuesGatherNd$userId_embedding/ExpandDims:output:00userId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:���������2)
'userId_embedding/to_sparse_input/values�
,userId_embedding/to_sparse_input/dense_shapeShape$userId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2.
,userId_embedding/to_sparse_input/dense_shape�
userId_embedding/valuesCast0userId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:���������2
userId_embedding/values�
5userId_embedding/userId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 27
5userId_embedding/userId_embedding_weights/Slice/begin�
4userId_embedding/userId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:26
4userId_embedding/userId_embedding_weights/Slice/size�
/userId_embedding/userId_embedding_weights/SliceSlice5userId_embedding/to_sparse_input/dense_shape:output:0>userId_embedding/userId_embedding_weights/Slice/begin:output:0=userId_embedding/userId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:21
/userId_embedding/userId_embedding_weights/Slice�
/userId_embedding/userId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/userId_embedding/userId_embedding_weights/Const�
.userId_embedding/userId_embedding_weights/ProdProd8userId_embedding/userId_embedding_weights/Slice:output:08userId_embedding/userId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: 20
.userId_embedding/userId_embedding_weights/Prod�
:userId_embedding/userId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :2<
:userId_embedding/userId_embedding_weights/GatherV2/indices�
7userId_embedding/userId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7userId_embedding/userId_embedding_weights/GatherV2/axis�
2userId_embedding/userId_embedding_weights/GatherV2GatherV25userId_embedding/to_sparse_input/dense_shape:output:0CuserId_embedding/userId_embedding_weights/GatherV2/indices:output:0@userId_embedding/userId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 24
2userId_embedding/userId_embedding_weights/GatherV2�
0userId_embedding/userId_embedding_weights/Cast/xPack7userId_embedding/userId_embedding_weights/Prod:output:0;userId_embedding/userId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:22
0userId_embedding/userId_embedding_weights/Cast/x�
7userId_embedding/userId_embedding_weights/SparseReshapeSparseReshape0userId_embedding/to_sparse_input/indices:index:05userId_embedding/to_sparse_input/dense_shape:output:09userId_embedding/userId_embedding_weights/Cast/x:output:0*-
_output_shapes
:���������:29
7userId_embedding/userId_embedding_weights/SparseReshape�
@userId_embedding/userId_embedding_weights/SparseReshape/IdentityIdentityuserId_embedding/values:y:0*
T0	*#
_output_shapes
:���������2B
@userId_embedding/userId_embedding_weights/SparseReshape/Identity�
8userId_embedding/userId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2:
8userId_embedding/userId_embedding_weights/GreaterEqual/y�
6userId_embedding/userId_embedding_weights/GreaterEqualGreaterEqualIuserId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0AuserId_embedding/userId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:���������28
6userId_embedding/userId_embedding_weights/GreaterEqual�
/userId_embedding/userId_embedding_weights/WhereWhere:userId_embedding/userId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:���������21
/userId_embedding/userId_embedding_weights/Where�
7userId_embedding/userId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������29
7userId_embedding/userId_embedding_weights/Reshape/shape�
1userId_embedding/userId_embedding_weights/ReshapeReshape7userId_embedding/userId_embedding_weights/Where:index:0@userId_embedding/userId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:���������23
1userId_embedding/userId_embedding_weights/Reshape�
9userId_embedding/userId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9userId_embedding/userId_embedding_weights/GatherV2_1/axis�
4userId_embedding/userId_embedding_weights/GatherV2_1GatherV2HuserId_embedding/userId_embedding_weights/SparseReshape:output_indices:0:userId_embedding/userId_embedding_weights/Reshape:output:0BuserId_embedding/userId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:���������26
4userId_embedding/userId_embedding_weights/GatherV2_1�
9userId_embedding/userId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9userId_embedding/userId_embedding_weights/GatherV2_2/axis�
4userId_embedding/userId_embedding_weights/GatherV2_2GatherV2IuserId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0:userId_embedding/userId_embedding_weights/Reshape:output:0BuserId_embedding/userId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:���������26
4userId_embedding/userId_embedding_weights/GatherV2_2�
2userId_embedding/userId_embedding_weights/IdentityIdentityFuserId_embedding/userId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:24
2userId_embedding/userId_embedding_weights/Identity�
CuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 2E
CuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/Const�
QuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows=userId_embedding/userId_embedding_weights/GatherV2_1:output:0=userId_embedding/userId_embedding_weights/GatherV2_2:output:0;userId_embedding/userId_embedding_weights/Identity:output:0LuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:���������:���������:���������:���������2S
QuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows�
UuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2W
UuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack�
WuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2Y
WuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1�
WuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2Y
WuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2�
OuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicebuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0^userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0`userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0`userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2Q
OuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice�
HuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/UniqueUniqueauserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:���������:���������2J
HuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique�
RuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherYuserid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_212211LuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*l
_classb
`^loc:@userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/212211*'
_output_shapes
:���������
*
dtype02T
RuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup�
[userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity[userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*l
_classb
`^loc:@userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/212211*'
_output_shapes
:���������
2]
[userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity�
]userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1IdentityduserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������
2_
]userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1�
AuserId_embedding/userId_embedding_weights/embedding_lookup_sparseSparseSegmentMeanfuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0NuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:idx:0XuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:���������
2C
AuserId_embedding/userId_embedding_weights/embedding_lookup_sparse�
9userId_embedding/userId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2;
9userId_embedding/userId_embedding_weights/Reshape_1/shape�
3userId_embedding/userId_embedding_weights/Reshape_1ReshapeguserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0BuserId_embedding/userId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:���������25
3userId_embedding/userId_embedding_weights/Reshape_1�
/userId_embedding/userId_embedding_weights/ShapeShapeJuserId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:21
/userId_embedding/userId_embedding_weights/Shape�
=userId_embedding/userId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
=userId_embedding/userId_embedding_weights/strided_slice/stack�
?userId_embedding/userId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?userId_embedding/userId_embedding_weights/strided_slice/stack_1�
?userId_embedding/userId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?userId_embedding/userId_embedding_weights/strided_slice/stack_2�
7userId_embedding/userId_embedding_weights/strided_sliceStridedSlice8userId_embedding/userId_embedding_weights/Shape:output:0FuserId_embedding/userId_embedding_weights/strided_slice/stack:output:0HuserId_embedding/userId_embedding_weights/strided_slice/stack_1:output:0HuserId_embedding/userId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7userId_embedding/userId_embedding_weights/strided_slice�
1userId_embedding/userId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :23
1userId_embedding/userId_embedding_weights/stack/0�
/userId_embedding/userId_embedding_weights/stackPack:userId_embedding/userId_embedding_weights/stack/0:output:0@userId_embedding/userId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:21
/userId_embedding/userId_embedding_weights/stack�
.userId_embedding/userId_embedding_weights/TileTile<userId_embedding/userId_embedding_weights/Reshape_1:output:08userId_embedding/userId_embedding_weights/stack:output:0*
T0
*0
_output_shapes
:������������������20
.userId_embedding/userId_embedding_weights/Tile�
4userId_embedding/userId_embedding_weights/zeros_like	ZerosLikeJuserId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
26
4userId_embedding/userId_embedding_weights/zeros_like�
)userId_embedding/userId_embedding_weightsSelect7userId_embedding/userId_embedding_weights/Tile:output:08userId_embedding/userId_embedding_weights/zeros_like:y:0JuserId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
2+
)userId_embedding/userId_embedding_weights�
0userId_embedding/userId_embedding_weights/Cast_1Cast5userId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:22
0userId_embedding/userId_embedding_weights/Cast_1�
7userId_embedding/userId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 29
7userId_embedding/userId_embedding_weights/Slice_1/begin�
6userId_embedding/userId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:28
6userId_embedding/userId_embedding_weights/Slice_1/size�
1userId_embedding/userId_embedding_weights/Slice_1Slice4userId_embedding/userId_embedding_weights/Cast_1:y:0@userId_embedding/userId_embedding_weights/Slice_1/begin:output:0?userId_embedding/userId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:23
1userId_embedding/userId_embedding_weights/Slice_1�
1userId_embedding/userId_embedding_weights/Shape_1Shape2userId_embedding/userId_embedding_weights:output:0*
T0*
_output_shapes
:23
1userId_embedding/userId_embedding_weights/Shape_1�
7userId_embedding/userId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:29
7userId_embedding/userId_embedding_weights/Slice_2/begin�
6userId_embedding/userId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
���������28
6userId_embedding/userId_embedding_weights/Slice_2/size�
1userId_embedding/userId_embedding_weights/Slice_2Slice:userId_embedding/userId_embedding_weights/Shape_1:output:0@userId_embedding/userId_embedding_weights/Slice_2/begin:output:0?userId_embedding/userId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:23
1userId_embedding/userId_embedding_weights/Slice_2�
5userId_embedding/userId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5userId_embedding/userId_embedding_weights/concat/axis�
0userId_embedding/userId_embedding_weights/concatConcatV2:userId_embedding/userId_embedding_weights/Slice_1:output:0:userId_embedding/userId_embedding_weights/Slice_2:output:0>userId_embedding/userId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0userId_embedding/userId_embedding_weights/concat�
3userId_embedding/userId_embedding_weights/Reshape_2Reshape2userId_embedding/userId_embedding_weights:output:09userId_embedding/userId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:���������
25
3userId_embedding/userId_embedding_weights/Reshape_2�
userId_embedding/ShapeShape<userId_embedding/userId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:2
userId_embedding/Shape�
$userId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$userId_embedding/strided_slice/stack�
&userId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&userId_embedding/strided_slice/stack_1�
&userId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&userId_embedding/strided_slice/stack_2�
userId_embedding/strided_sliceStridedSliceuserId_embedding/Shape:output:0-userId_embedding/strided_slice/stack:output:0/userId_embedding/strided_slice/stack_1:output:0/userId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
userId_embedding/strided_slice�
 userId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2"
 userId_embedding/Reshape/shape/1�
userId_embedding/Reshape/shapePack'userId_embedding/strided_slice:output:0)userId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
userId_embedding/Reshape/shape�
userId_embedding/ReshapeReshape<userId_embedding/userId_embedding_weights/Reshape_2:output:0'userId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:���������
2
userId_embedding/Reshapeq
concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat/concat_dim
concat/concatIdentity!userId_embedding/Reshape:output:0*
T0*'
_output_shapes
:���������
2
concat/concatj
IdentityIdentityconcat/concat:output:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*5
_input_shapes$
":���������:���������::M I
#
_output_shapes
:���������
"
_user_specified_name
features:MI
#
_output_shapes
:���������
"
_user_specified_name
features
�
�
1__inference_dense_features_1_layer_call_fn_213397
features_movieid
features_userid
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallfeatures_movieidfeatures_useridunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_dense_features_1_layer_call_and_return_conditional_losses_2122512
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*5
_input_shapes$
":���������:���������:22
StatefulPartitionedCallStatefulPartitionedCall:U Q
#
_output_shapes
:���������
*
_user_specified_namefeatures/movieId:TP
#
_output_shapes
:���������
)
_user_specified_namefeatures/userId
�
�
H__inference_functional_1_layer_call_and_return_conditional_losses_212562

inputs
inputs_1
dense_features_212539
dense_features_1_212542
dense_212546
dense_212548
dense_1_212551
dense_1_212553
dense_2_212556
dense_2_212558
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�&dense_features/StatefulPartitionedCall�(dense_features_1/StatefulPartitionedCall�
&dense_features/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1dense_features_212539*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_2121452(
&dense_features/StatefulPartitionedCall�
(dense_features_1/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1dense_features_1_212542*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_dense_features_1_layer_call_and_return_conditional_losses_2123362*
(dense_features_1/StatefulPartitionedCall�
concatenate/PartitionedCallPartitionedCall/dense_features/StatefulPartitionedCall:output:01dense_features_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_2123642
concatenate/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_212546dense_212548*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2123842
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_212551dense_1_212553*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2124112!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_212556dense_2_212558*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2124382!
dense_2/StatefulPartitionedCall�
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall'^dense_features/StatefulPartitionedCall)^dense_features_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:���������:���������::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2P
&dense_features/StatefulPartitionedCall&dense_features/StatefulPartitionedCall2T
(dense_features_1/StatefulPartitionedCall(dense_features_1/StatefulPartitionedCall:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
-__inference_functional_1_layer_call_fn_212532
movieid

userid
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallmovieiduseridunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_2125132
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:���������:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	movieId:KG
#
_output_shapes
:���������
 
_user_specified_nameuserId
�
q
G__inference_concatenate_layer_call_and_return_conditional_losses_212364

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������
:���������
:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs:OK
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
/__inference_dense_features_layer_call_fn_213219
features_movieid
features_userid
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallfeatures_movieidfeatures_useridunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_2121452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*5
_input_shapes$
":���������:���������:22
StatefulPartitionedCallStatefulPartitionedCall:U Q
#
_output_shapes
:���������
*
_user_specified_namefeatures/movieId:TP
#
_output_shapes
:���������
)
_user_specified_namefeatures/userId
��
�
H__inference_functional_1_layer_call_and_return_conditional_losses_212801
inputs_movieid
inputs_useridn
jdense_features_movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_212658n
jdense_features_1_userid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_212738(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity��
/dense_features/movieId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������21
/dense_features/movieId_embedding/ExpandDims/dim�
+dense_features/movieId_embedding/ExpandDims
ExpandDimsinputs_movieid8dense_features/movieId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������2-
+dense_features/movieId_embedding/ExpandDims�
?dense_features/movieId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
���������2A
?dense_features/movieId_embedding/to_sparse_input/ignore_value/x�
9dense_features/movieId_embedding/to_sparse_input/NotEqualNotEqual4dense_features/movieId_embedding/ExpandDims:output:0Hdense_features/movieId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:���������2;
9dense_features/movieId_embedding/to_sparse_input/NotEqual�
8dense_features/movieId_embedding/to_sparse_input/indicesWhere=dense_features/movieId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:���������2:
8dense_features/movieId_embedding/to_sparse_input/indices�
7dense_features/movieId_embedding/to_sparse_input/valuesGatherNd4dense_features/movieId_embedding/ExpandDims:output:0@dense_features/movieId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:���������29
7dense_features/movieId_embedding/to_sparse_input/values�
<dense_features/movieId_embedding/to_sparse_input/dense_shapeShape4dense_features/movieId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2>
<dense_features/movieId_embedding/to_sparse_input/dense_shape�
'dense_features/movieId_embedding/valuesCast@dense_features/movieId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:���������2)
'dense_features/movieId_embedding/values�
Fdense_features/movieId_embedding/movieId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 2H
Fdense_features/movieId_embedding/movieId_embedding_weights/Slice/begin�
Edense_features/movieId_embedding/movieId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:2G
Edense_features/movieId_embedding/movieId_embedding_weights/Slice/size�
@dense_features/movieId_embedding/movieId_embedding_weights/SliceSliceEdense_features/movieId_embedding/to_sparse_input/dense_shape:output:0Odense_features/movieId_embedding/movieId_embedding_weights/Slice/begin:output:0Ndense_features/movieId_embedding/movieId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:2B
@dense_features/movieId_embedding/movieId_embedding_weights/Slice�
@dense_features/movieId_embedding/movieId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2B
@dense_features/movieId_embedding/movieId_embedding_weights/Const�
?dense_features/movieId_embedding/movieId_embedding_weights/ProdProdIdense_features/movieId_embedding/movieId_embedding_weights/Slice:output:0Idense_features/movieId_embedding/movieId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: 2A
?dense_features/movieId_embedding/movieId_embedding_weights/Prod�
Kdense_features/movieId_embedding/movieId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :2M
Kdense_features/movieId_embedding/movieId_embedding_weights/GatherV2/indices�
Hdense_features/movieId_embedding/movieId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hdense_features/movieId_embedding/movieId_embedding_weights/GatherV2/axis�
Cdense_features/movieId_embedding/movieId_embedding_weights/GatherV2GatherV2Edense_features/movieId_embedding/to_sparse_input/dense_shape:output:0Tdense_features/movieId_embedding/movieId_embedding_weights/GatherV2/indices:output:0Qdense_features/movieId_embedding/movieId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 2E
Cdense_features/movieId_embedding/movieId_embedding_weights/GatherV2�
Adense_features/movieId_embedding/movieId_embedding_weights/Cast/xPackHdense_features/movieId_embedding/movieId_embedding_weights/Prod:output:0Ldense_features/movieId_embedding/movieId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:2C
Adense_features/movieId_embedding/movieId_embedding_weights/Cast/x�
Hdense_features/movieId_embedding/movieId_embedding_weights/SparseReshapeSparseReshape@dense_features/movieId_embedding/to_sparse_input/indices:index:0Edense_features/movieId_embedding/to_sparse_input/dense_shape:output:0Jdense_features/movieId_embedding/movieId_embedding_weights/Cast/x:output:0*-
_output_shapes
:���������:2J
Hdense_features/movieId_embedding/movieId_embedding_weights/SparseReshape�
Qdense_features/movieId_embedding/movieId_embedding_weights/SparseReshape/IdentityIdentity+dense_features/movieId_embedding/values:y:0*
T0	*#
_output_shapes
:���������2S
Qdense_features/movieId_embedding/movieId_embedding_weights/SparseReshape/Identity�
Idense_features/movieId_embedding/movieId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2K
Idense_features/movieId_embedding/movieId_embedding_weights/GreaterEqual/y�
Gdense_features/movieId_embedding/movieId_embedding_weights/GreaterEqualGreaterEqualZdense_features/movieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0Rdense_features/movieId_embedding/movieId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:���������2I
Gdense_features/movieId_embedding/movieId_embedding_weights/GreaterEqual�
@dense_features/movieId_embedding/movieId_embedding_weights/WhereWhereKdense_features/movieId_embedding/movieId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:���������2B
@dense_features/movieId_embedding/movieId_embedding_weights/Where�
Hdense_features/movieId_embedding/movieId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������2J
Hdense_features/movieId_embedding/movieId_embedding_weights/Reshape/shape�
Bdense_features/movieId_embedding/movieId_embedding_weights/ReshapeReshapeHdense_features/movieId_embedding/movieId_embedding_weights/Where:index:0Qdense_features/movieId_embedding/movieId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:���������2D
Bdense_features/movieId_embedding/movieId_embedding_weights/Reshape�
Jdense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jdense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1/axis�
Edense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1GatherV2Ydense_features/movieId_embedding/movieId_embedding_weights/SparseReshape:output_indices:0Kdense_features/movieId_embedding/movieId_embedding_weights/Reshape:output:0Sdense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:���������2G
Edense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1�
Jdense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jdense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2/axis�
Edense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2GatherV2Zdense_features/movieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0Kdense_features/movieId_embedding/movieId_embedding_weights/Reshape:output:0Sdense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:���������2G
Edense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2�
Cdense_features/movieId_embedding/movieId_embedding_weights/IdentityIdentityWdense_features/movieId_embedding/movieId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:2E
Cdense_features/movieId_embedding/movieId_embedding_weights/Identity�
Tdense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 2V
Tdense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/Const�
bdense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsNdense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1:output:0Ndense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2:output:0Ldense_features/movieId_embedding/movieId_embedding_weights/Identity:output:0]dense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:���������:���������:���������:���������2d
bdense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows�
fdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2h
fdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack�
hdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2j
hdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1�
hdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2j
hdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2�
`dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicesdense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0odense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0qdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0qdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2b
`dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice�
Ydense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/UniqueUniquerdense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:���������:���������2[
Ydense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique�
cdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherjdense_features_movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_212658]dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*}
_classs
qoloc:@dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/212658*'
_output_shapes
:���������
*
dtype02e
cdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup�
ldense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityldense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*}
_classs
qoloc:@dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/212658*'
_output_shapes
:���������
2n
ldense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity�
ndense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1Identityudense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������
2p
ndense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1�
Rdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparseSparseSegmentMeanwdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0_dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:idx:0idense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:���������
2T
Rdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse�
Jdense_features/movieId_embedding/movieId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2L
Jdense_features/movieId_embedding/movieId_embedding_weights/Reshape_1/shape�
Ddense_features/movieId_embedding/movieId_embedding_weights/Reshape_1Reshapexdense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0Sdense_features/movieId_embedding/movieId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:���������2F
Ddense_features/movieId_embedding/movieId_embedding_weights/Reshape_1�
@dense_features/movieId_embedding/movieId_embedding_weights/ShapeShape[dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:2B
@dense_features/movieId_embedding/movieId_embedding_weights/Shape�
Ndense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2P
Ndense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack�
Pdense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2R
Pdense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_1�
Pdense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2R
Pdense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_2�
Hdense_features/movieId_embedding/movieId_embedding_weights/strided_sliceStridedSliceIdense_features/movieId_embedding/movieId_embedding_weights/Shape:output:0Wdense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack:output:0Ydense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_1:output:0Ydense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2J
Hdense_features/movieId_embedding/movieId_embedding_weights/strided_slice�
Bdense_features/movieId_embedding/movieId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :2D
Bdense_features/movieId_embedding/movieId_embedding_weights/stack/0�
@dense_features/movieId_embedding/movieId_embedding_weights/stackPackKdense_features/movieId_embedding/movieId_embedding_weights/stack/0:output:0Qdense_features/movieId_embedding/movieId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:2B
@dense_features/movieId_embedding/movieId_embedding_weights/stack�
?dense_features/movieId_embedding/movieId_embedding_weights/TileTileMdense_features/movieId_embedding/movieId_embedding_weights/Reshape_1:output:0Idense_features/movieId_embedding/movieId_embedding_weights/stack:output:0*
T0
*0
_output_shapes
:������������������2A
?dense_features/movieId_embedding/movieId_embedding_weights/Tile�
Edense_features/movieId_embedding/movieId_embedding_weights/zeros_like	ZerosLike[dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
2G
Edense_features/movieId_embedding/movieId_embedding_weights/zeros_like�
:dense_features/movieId_embedding/movieId_embedding_weightsSelectHdense_features/movieId_embedding/movieId_embedding_weights/Tile:output:0Idense_features/movieId_embedding/movieId_embedding_weights/zeros_like:y:0[dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
2<
:dense_features/movieId_embedding/movieId_embedding_weights�
Adense_features/movieId_embedding/movieId_embedding_weights/Cast_1CastEdense_features/movieId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:2C
Adense_features/movieId_embedding/movieId_embedding_weights/Cast_1�
Hdense_features/movieId_embedding/movieId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 2J
Hdense_features/movieId_embedding/movieId_embedding_weights/Slice_1/begin�
Gdense_features/movieId_embedding/movieId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:2I
Gdense_features/movieId_embedding/movieId_embedding_weights/Slice_1/size�
Bdense_features/movieId_embedding/movieId_embedding_weights/Slice_1SliceEdense_features/movieId_embedding/movieId_embedding_weights/Cast_1:y:0Qdense_features/movieId_embedding/movieId_embedding_weights/Slice_1/begin:output:0Pdense_features/movieId_embedding/movieId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:2D
Bdense_features/movieId_embedding/movieId_embedding_weights/Slice_1�
Bdense_features/movieId_embedding/movieId_embedding_weights/Shape_1ShapeCdense_features/movieId_embedding/movieId_embedding_weights:output:0*
T0*
_output_shapes
:2D
Bdense_features/movieId_embedding/movieId_embedding_weights/Shape_1�
Hdense_features/movieId_embedding/movieId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:2J
Hdense_features/movieId_embedding/movieId_embedding_weights/Slice_2/begin�
Gdense_features/movieId_embedding/movieId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
���������2I
Gdense_features/movieId_embedding/movieId_embedding_weights/Slice_2/size�
Bdense_features/movieId_embedding/movieId_embedding_weights/Slice_2SliceKdense_features/movieId_embedding/movieId_embedding_weights/Shape_1:output:0Qdense_features/movieId_embedding/movieId_embedding_weights/Slice_2/begin:output:0Pdense_features/movieId_embedding/movieId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:2D
Bdense_features/movieId_embedding/movieId_embedding_weights/Slice_2�
Fdense_features/movieId_embedding/movieId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fdense_features/movieId_embedding/movieId_embedding_weights/concat/axis�
Adense_features/movieId_embedding/movieId_embedding_weights/concatConcatV2Kdense_features/movieId_embedding/movieId_embedding_weights/Slice_1:output:0Kdense_features/movieId_embedding/movieId_embedding_weights/Slice_2:output:0Odense_features/movieId_embedding/movieId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:2C
Adense_features/movieId_embedding/movieId_embedding_weights/concat�
Ddense_features/movieId_embedding/movieId_embedding_weights/Reshape_2ReshapeCdense_features/movieId_embedding/movieId_embedding_weights:output:0Jdense_features/movieId_embedding/movieId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:���������
2F
Ddense_features/movieId_embedding/movieId_embedding_weights/Reshape_2�
&dense_features/movieId_embedding/ShapeShapeMdense_features/movieId_embedding/movieId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:2(
&dense_features/movieId_embedding/Shape�
4dense_features/movieId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4dense_features/movieId_embedding/strided_slice/stack�
6dense_features/movieId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6dense_features/movieId_embedding/strided_slice/stack_1�
6dense_features/movieId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6dense_features/movieId_embedding/strided_slice/stack_2�
.dense_features/movieId_embedding/strided_sliceStridedSlice/dense_features/movieId_embedding/Shape:output:0=dense_features/movieId_embedding/strided_slice/stack:output:0?dense_features/movieId_embedding/strided_slice/stack_1:output:0?dense_features/movieId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.dense_features/movieId_embedding/strided_slice�
0dense_features/movieId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
22
0dense_features/movieId_embedding/Reshape/shape/1�
.dense_features/movieId_embedding/Reshape/shapePack7dense_features/movieId_embedding/strided_slice:output:09dense_features/movieId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:20
.dense_features/movieId_embedding/Reshape/shape�
(dense_features/movieId_embedding/ReshapeReshapeMdense_features/movieId_embedding/movieId_embedding_weights/Reshape_2:output:07dense_features/movieId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:���������
2*
(dense_features/movieId_embedding/Reshape�
 dense_features/concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 dense_features/concat/concat_dim�
dense_features/concat/concatIdentity1dense_features/movieId_embedding/Reshape:output:0*
T0*'
_output_shapes
:���������
2
dense_features/concat/concat�
0dense_features_1/userId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������22
0dense_features_1/userId_embedding/ExpandDims/dim�
,dense_features_1/userId_embedding/ExpandDims
ExpandDimsinputs_userid9dense_features_1/userId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������2.
,dense_features_1/userId_embedding/ExpandDims�
@dense_features_1/userId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
���������2B
@dense_features_1/userId_embedding/to_sparse_input/ignore_value/x�
:dense_features_1/userId_embedding/to_sparse_input/NotEqualNotEqual5dense_features_1/userId_embedding/ExpandDims:output:0Idense_features_1/userId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:���������2<
:dense_features_1/userId_embedding/to_sparse_input/NotEqual�
9dense_features_1/userId_embedding/to_sparse_input/indicesWhere>dense_features_1/userId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:���������2;
9dense_features_1/userId_embedding/to_sparse_input/indices�
8dense_features_1/userId_embedding/to_sparse_input/valuesGatherNd5dense_features_1/userId_embedding/ExpandDims:output:0Adense_features_1/userId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:���������2:
8dense_features_1/userId_embedding/to_sparse_input/values�
=dense_features_1/userId_embedding/to_sparse_input/dense_shapeShape5dense_features_1/userId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2?
=dense_features_1/userId_embedding/to_sparse_input/dense_shape�
(dense_features_1/userId_embedding/valuesCastAdense_features_1/userId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:���������2*
(dense_features_1/userId_embedding/values�
Fdense_features_1/userId_embedding/userId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 2H
Fdense_features_1/userId_embedding/userId_embedding_weights/Slice/begin�
Edense_features_1/userId_embedding/userId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:2G
Edense_features_1/userId_embedding/userId_embedding_weights/Slice/size�
@dense_features_1/userId_embedding/userId_embedding_weights/SliceSliceFdense_features_1/userId_embedding/to_sparse_input/dense_shape:output:0Odense_features_1/userId_embedding/userId_embedding_weights/Slice/begin:output:0Ndense_features_1/userId_embedding/userId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:2B
@dense_features_1/userId_embedding/userId_embedding_weights/Slice�
@dense_features_1/userId_embedding/userId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2B
@dense_features_1/userId_embedding/userId_embedding_weights/Const�
?dense_features_1/userId_embedding/userId_embedding_weights/ProdProdIdense_features_1/userId_embedding/userId_embedding_weights/Slice:output:0Idense_features_1/userId_embedding/userId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: 2A
?dense_features_1/userId_embedding/userId_embedding_weights/Prod�
Kdense_features_1/userId_embedding/userId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :2M
Kdense_features_1/userId_embedding/userId_embedding_weights/GatherV2/indices�
Hdense_features_1/userId_embedding/userId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hdense_features_1/userId_embedding/userId_embedding_weights/GatherV2/axis�
Cdense_features_1/userId_embedding/userId_embedding_weights/GatherV2GatherV2Fdense_features_1/userId_embedding/to_sparse_input/dense_shape:output:0Tdense_features_1/userId_embedding/userId_embedding_weights/GatherV2/indices:output:0Qdense_features_1/userId_embedding/userId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 2E
Cdense_features_1/userId_embedding/userId_embedding_weights/GatherV2�
Adense_features_1/userId_embedding/userId_embedding_weights/Cast/xPackHdense_features_1/userId_embedding/userId_embedding_weights/Prod:output:0Ldense_features_1/userId_embedding/userId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:2C
Adense_features_1/userId_embedding/userId_embedding_weights/Cast/x�
Hdense_features_1/userId_embedding/userId_embedding_weights/SparseReshapeSparseReshapeAdense_features_1/userId_embedding/to_sparse_input/indices:index:0Fdense_features_1/userId_embedding/to_sparse_input/dense_shape:output:0Jdense_features_1/userId_embedding/userId_embedding_weights/Cast/x:output:0*-
_output_shapes
:���������:2J
Hdense_features_1/userId_embedding/userId_embedding_weights/SparseReshape�
Qdense_features_1/userId_embedding/userId_embedding_weights/SparseReshape/IdentityIdentity,dense_features_1/userId_embedding/values:y:0*
T0	*#
_output_shapes
:���������2S
Qdense_features_1/userId_embedding/userId_embedding_weights/SparseReshape/Identity�
Idense_features_1/userId_embedding/userId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2K
Idense_features_1/userId_embedding/userId_embedding_weights/GreaterEqual/y�
Gdense_features_1/userId_embedding/userId_embedding_weights/GreaterEqualGreaterEqualZdense_features_1/userId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0Rdense_features_1/userId_embedding/userId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:���������2I
Gdense_features_1/userId_embedding/userId_embedding_weights/GreaterEqual�
@dense_features_1/userId_embedding/userId_embedding_weights/WhereWhereKdense_features_1/userId_embedding/userId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:���������2B
@dense_features_1/userId_embedding/userId_embedding_weights/Where�
Hdense_features_1/userId_embedding/userId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������2J
Hdense_features_1/userId_embedding/userId_embedding_weights/Reshape/shape�
Bdense_features_1/userId_embedding/userId_embedding_weights/ReshapeReshapeHdense_features_1/userId_embedding/userId_embedding_weights/Where:index:0Qdense_features_1/userId_embedding/userId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:���������2D
Bdense_features_1/userId_embedding/userId_embedding_weights/Reshape�
Jdense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jdense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1/axis�
Edense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1GatherV2Ydense_features_1/userId_embedding/userId_embedding_weights/SparseReshape:output_indices:0Kdense_features_1/userId_embedding/userId_embedding_weights/Reshape:output:0Sdense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:���������2G
Edense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1�
Jdense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jdense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2/axis�
Edense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2GatherV2Zdense_features_1/userId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0Kdense_features_1/userId_embedding/userId_embedding_weights/Reshape:output:0Sdense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:���������2G
Edense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2�
Cdense_features_1/userId_embedding/userId_embedding_weights/IdentityIdentityWdense_features_1/userId_embedding/userId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:2E
Cdense_features_1/userId_embedding/userId_embedding_weights/Identity�
Tdense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 2V
Tdense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/Const�
bdense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsNdense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1:output:0Ndense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2:output:0Ldense_features_1/userId_embedding/userId_embedding_weights/Identity:output:0]dense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:���������:���������:���������:���������2d
bdense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows�
fdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2h
fdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack�
hdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2j
hdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1�
hdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2j
hdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2�
`dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicesdense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0odense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0qdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0qdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2b
`dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice�
Ydense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/UniqueUniquerdense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:���������:���������2[
Ydense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique�
cdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherjdense_features_1_userid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_212738]dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*}
_classs
qoloc:@dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/212738*'
_output_shapes
:���������
*
dtype02e
cdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup�
ldense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityldense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*}
_classs
qoloc:@dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/212738*'
_output_shapes
:���������
2n
ldense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity�
ndense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1Identityudense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������
2p
ndense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1�
Rdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparseSparseSegmentMeanwdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0_dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:idx:0idense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:���������
2T
Rdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse�
Jdense_features_1/userId_embedding/userId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2L
Jdense_features_1/userId_embedding/userId_embedding_weights/Reshape_1/shape�
Ddense_features_1/userId_embedding/userId_embedding_weights/Reshape_1Reshapexdense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0Sdense_features_1/userId_embedding/userId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:���������2F
Ddense_features_1/userId_embedding/userId_embedding_weights/Reshape_1�
@dense_features_1/userId_embedding/userId_embedding_weights/ShapeShape[dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:2B
@dense_features_1/userId_embedding/userId_embedding_weights/Shape�
Ndense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2P
Ndense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack�
Pdense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2R
Pdense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_1�
Pdense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2R
Pdense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_2�
Hdense_features_1/userId_embedding/userId_embedding_weights/strided_sliceStridedSliceIdense_features_1/userId_embedding/userId_embedding_weights/Shape:output:0Wdense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack:output:0Ydense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_1:output:0Ydense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2J
Hdense_features_1/userId_embedding/userId_embedding_weights/strided_slice�
Bdense_features_1/userId_embedding/userId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :2D
Bdense_features_1/userId_embedding/userId_embedding_weights/stack/0�
@dense_features_1/userId_embedding/userId_embedding_weights/stackPackKdense_features_1/userId_embedding/userId_embedding_weights/stack/0:output:0Qdense_features_1/userId_embedding/userId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:2B
@dense_features_1/userId_embedding/userId_embedding_weights/stack�
?dense_features_1/userId_embedding/userId_embedding_weights/TileTileMdense_features_1/userId_embedding/userId_embedding_weights/Reshape_1:output:0Idense_features_1/userId_embedding/userId_embedding_weights/stack:output:0*
T0
*0
_output_shapes
:������������������2A
?dense_features_1/userId_embedding/userId_embedding_weights/Tile�
Edense_features_1/userId_embedding/userId_embedding_weights/zeros_like	ZerosLike[dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
2G
Edense_features_1/userId_embedding/userId_embedding_weights/zeros_like�
:dense_features_1/userId_embedding/userId_embedding_weightsSelectHdense_features_1/userId_embedding/userId_embedding_weights/Tile:output:0Idense_features_1/userId_embedding/userId_embedding_weights/zeros_like:y:0[dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
2<
:dense_features_1/userId_embedding/userId_embedding_weights�
Adense_features_1/userId_embedding/userId_embedding_weights/Cast_1CastFdense_features_1/userId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:2C
Adense_features_1/userId_embedding/userId_embedding_weights/Cast_1�
Hdense_features_1/userId_embedding/userId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 2J
Hdense_features_1/userId_embedding/userId_embedding_weights/Slice_1/begin�
Gdense_features_1/userId_embedding/userId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:2I
Gdense_features_1/userId_embedding/userId_embedding_weights/Slice_1/size�
Bdense_features_1/userId_embedding/userId_embedding_weights/Slice_1SliceEdense_features_1/userId_embedding/userId_embedding_weights/Cast_1:y:0Qdense_features_1/userId_embedding/userId_embedding_weights/Slice_1/begin:output:0Pdense_features_1/userId_embedding/userId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:2D
Bdense_features_1/userId_embedding/userId_embedding_weights/Slice_1�
Bdense_features_1/userId_embedding/userId_embedding_weights/Shape_1ShapeCdense_features_1/userId_embedding/userId_embedding_weights:output:0*
T0*
_output_shapes
:2D
Bdense_features_1/userId_embedding/userId_embedding_weights/Shape_1�
Hdense_features_1/userId_embedding/userId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:2J
Hdense_features_1/userId_embedding/userId_embedding_weights/Slice_2/begin�
Gdense_features_1/userId_embedding/userId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
���������2I
Gdense_features_1/userId_embedding/userId_embedding_weights/Slice_2/size�
Bdense_features_1/userId_embedding/userId_embedding_weights/Slice_2SliceKdense_features_1/userId_embedding/userId_embedding_weights/Shape_1:output:0Qdense_features_1/userId_embedding/userId_embedding_weights/Slice_2/begin:output:0Pdense_features_1/userId_embedding/userId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:2D
Bdense_features_1/userId_embedding/userId_embedding_weights/Slice_2�
Fdense_features_1/userId_embedding/userId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fdense_features_1/userId_embedding/userId_embedding_weights/concat/axis�
Adense_features_1/userId_embedding/userId_embedding_weights/concatConcatV2Kdense_features_1/userId_embedding/userId_embedding_weights/Slice_1:output:0Kdense_features_1/userId_embedding/userId_embedding_weights/Slice_2:output:0Odense_features_1/userId_embedding/userId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:2C
Adense_features_1/userId_embedding/userId_embedding_weights/concat�
Ddense_features_1/userId_embedding/userId_embedding_weights/Reshape_2ReshapeCdense_features_1/userId_embedding/userId_embedding_weights:output:0Jdense_features_1/userId_embedding/userId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:���������
2F
Ddense_features_1/userId_embedding/userId_embedding_weights/Reshape_2�
'dense_features_1/userId_embedding/ShapeShapeMdense_features_1/userId_embedding/userId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:2)
'dense_features_1/userId_embedding/Shape�
5dense_features_1/userId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5dense_features_1/userId_embedding/strided_slice/stack�
7dense_features_1/userId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/userId_embedding/strided_slice/stack_1�
7dense_features_1/userId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/userId_embedding/strided_slice/stack_2�
/dense_features_1/userId_embedding/strided_sliceStridedSlice0dense_features_1/userId_embedding/Shape:output:0>dense_features_1/userId_embedding/strided_slice/stack:output:0@dense_features_1/userId_embedding/strided_slice/stack_1:output:0@dense_features_1/userId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/dense_features_1/userId_embedding/strided_slice�
1dense_features_1/userId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
23
1dense_features_1/userId_embedding/Reshape/shape/1�
/dense_features_1/userId_embedding/Reshape/shapePack8dense_features_1/userId_embedding/strided_slice:output:0:dense_features_1/userId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:21
/dense_features_1/userId_embedding/Reshape/shape�
)dense_features_1/userId_embedding/ReshapeReshapeMdense_features_1/userId_embedding/userId_embedding_weights/Reshape_2:output:08dense_features_1/userId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:���������
2+
)dense_features_1/userId_embedding/Reshape�
"dense_features_1/concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"dense_features_1/concat/concat_dim�
dense_features_1/concat/concatIdentity2dense_features_1/userId_embedding/Reshape:output:0*
T0*'
_output_shapes
:���������
2 
dense_features_1/concat/concatt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis�
concatenate/concatConcatV2%dense_features/concat/concat:output:0'dense_features_1/concat/concat:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatenate/concat�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2

dense/Relu�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
dense_1/Relu�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_2/Sigmoidg
IdentityIdentitydense_2/Sigmoid:y:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:���������:���������:::::::::S O
#
_output_shapes
:���������
(
_user_specified_nameinputs/movieId:RN
#
_output_shapes
:���������
'
_user_specified_nameinputs/userId
�
�
C__inference_dense_1_layer_call_and_return_conditional_losses_213449

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������
2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
:::O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
}
(__inference_dense_1_layer_call_fn_213458

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2124112
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
X
,__inference_concatenate_layer_call_fn_213418
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_2123642
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������
:���������
:Q M
'
_output_shapes
:���������

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������

"
_user_specified_name
inputs/1
�
}
(__inference_dense_2_layer_call_fn_213478

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2124382
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
J__inference_dense_features_layer_call_and_return_conditional_losses_212060
features

features_1_
[movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_212020
identity��
 movieId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 movieId_embedding/ExpandDims/dim�
movieId_embedding/ExpandDims
ExpandDimsfeatures)movieId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������2
movieId_embedding/ExpandDims�
0movieId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
���������22
0movieId_embedding/to_sparse_input/ignore_value/x�
*movieId_embedding/to_sparse_input/NotEqualNotEqual%movieId_embedding/ExpandDims:output:09movieId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:���������2,
*movieId_embedding/to_sparse_input/NotEqual�
)movieId_embedding/to_sparse_input/indicesWhere.movieId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:���������2+
)movieId_embedding/to_sparse_input/indices�
(movieId_embedding/to_sparse_input/valuesGatherNd%movieId_embedding/ExpandDims:output:01movieId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:���������2*
(movieId_embedding/to_sparse_input/values�
-movieId_embedding/to_sparse_input/dense_shapeShape%movieId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2/
-movieId_embedding/to_sparse_input/dense_shape�
movieId_embedding/valuesCast1movieId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:���������2
movieId_embedding/values�
7movieId_embedding/movieId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 29
7movieId_embedding/movieId_embedding_weights/Slice/begin�
6movieId_embedding/movieId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:28
6movieId_embedding/movieId_embedding_weights/Slice/size�
1movieId_embedding/movieId_embedding_weights/SliceSlice6movieId_embedding/to_sparse_input/dense_shape:output:0@movieId_embedding/movieId_embedding_weights/Slice/begin:output:0?movieId_embedding/movieId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:23
1movieId_embedding/movieId_embedding_weights/Slice�
1movieId_embedding/movieId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1movieId_embedding/movieId_embedding_weights/Const�
0movieId_embedding/movieId_embedding_weights/ProdProd:movieId_embedding/movieId_embedding_weights/Slice:output:0:movieId_embedding/movieId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: 22
0movieId_embedding/movieId_embedding_weights/Prod�
<movieId_embedding/movieId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :2>
<movieId_embedding/movieId_embedding_weights/GatherV2/indices�
9movieId_embedding/movieId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9movieId_embedding/movieId_embedding_weights/GatherV2/axis�
4movieId_embedding/movieId_embedding_weights/GatherV2GatherV26movieId_embedding/to_sparse_input/dense_shape:output:0EmovieId_embedding/movieId_embedding_weights/GatherV2/indices:output:0BmovieId_embedding/movieId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 26
4movieId_embedding/movieId_embedding_weights/GatherV2�
2movieId_embedding/movieId_embedding_weights/Cast/xPack9movieId_embedding/movieId_embedding_weights/Prod:output:0=movieId_embedding/movieId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:24
2movieId_embedding/movieId_embedding_weights/Cast/x�
9movieId_embedding/movieId_embedding_weights/SparseReshapeSparseReshape1movieId_embedding/to_sparse_input/indices:index:06movieId_embedding/to_sparse_input/dense_shape:output:0;movieId_embedding/movieId_embedding_weights/Cast/x:output:0*-
_output_shapes
:���������:2;
9movieId_embedding/movieId_embedding_weights/SparseReshape�
BmovieId_embedding/movieId_embedding_weights/SparseReshape/IdentityIdentitymovieId_embedding/values:y:0*
T0	*#
_output_shapes
:���������2D
BmovieId_embedding/movieId_embedding_weights/SparseReshape/Identity�
:movieId_embedding/movieId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2<
:movieId_embedding/movieId_embedding_weights/GreaterEqual/y�
8movieId_embedding/movieId_embedding_weights/GreaterEqualGreaterEqualKmovieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0CmovieId_embedding/movieId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:���������2:
8movieId_embedding/movieId_embedding_weights/GreaterEqual�
1movieId_embedding/movieId_embedding_weights/WhereWhere<movieId_embedding/movieId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:���������23
1movieId_embedding/movieId_embedding_weights/Where�
9movieId_embedding/movieId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������2;
9movieId_embedding/movieId_embedding_weights/Reshape/shape�
3movieId_embedding/movieId_embedding_weights/ReshapeReshape9movieId_embedding/movieId_embedding_weights/Where:index:0BmovieId_embedding/movieId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:���������25
3movieId_embedding/movieId_embedding_weights/Reshape�
;movieId_embedding/movieId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;movieId_embedding/movieId_embedding_weights/GatherV2_1/axis�
6movieId_embedding/movieId_embedding_weights/GatherV2_1GatherV2JmovieId_embedding/movieId_embedding_weights/SparseReshape:output_indices:0<movieId_embedding/movieId_embedding_weights/Reshape:output:0DmovieId_embedding/movieId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:���������28
6movieId_embedding/movieId_embedding_weights/GatherV2_1�
;movieId_embedding/movieId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;movieId_embedding/movieId_embedding_weights/GatherV2_2/axis�
6movieId_embedding/movieId_embedding_weights/GatherV2_2GatherV2KmovieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0<movieId_embedding/movieId_embedding_weights/Reshape:output:0DmovieId_embedding/movieId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:���������28
6movieId_embedding/movieId_embedding_weights/GatherV2_2�
4movieId_embedding/movieId_embedding_weights/IdentityIdentityHmovieId_embedding/movieId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:26
4movieId_embedding/movieId_embedding_weights/Identity�
EmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 2G
EmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/Const�
SmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows?movieId_embedding/movieId_embedding_weights/GatherV2_1:output:0?movieId_embedding/movieId_embedding_weights/GatherV2_2:output:0=movieId_embedding/movieId_embedding_weights/Identity:output:0NmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:���������:���������:���������:���������2U
SmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows�
WmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2Y
WmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack�
YmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2[
YmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1�
YmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2[
YmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2�
QmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicedmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0`movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0bmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0bmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2S
QmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice�
JmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/UniqueUniquecmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:���������:���������2L
JmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique�
TmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGather[movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_212020NmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*n
_classd
b`loc:@movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/212020*'
_output_shapes
:���������
*
dtype02V
TmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup�
]movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity]movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*n
_classd
b`loc:@movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/212020*'
_output_shapes
:���������
2_
]movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity�
_movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1IdentityfmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������
2a
_movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1�
CmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparseSparseSegmentMeanhmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0PmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:idx:0ZmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:���������
2E
CmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse�
;movieId_embedding/movieId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2=
;movieId_embedding/movieId_embedding_weights/Reshape_1/shape�
5movieId_embedding/movieId_embedding_weights/Reshape_1ReshapeimovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0DmovieId_embedding/movieId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:���������27
5movieId_embedding/movieId_embedding_weights/Reshape_1�
1movieId_embedding/movieId_embedding_weights/ShapeShapeLmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:23
1movieId_embedding/movieId_embedding_weights/Shape�
?movieId_embedding/movieId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2A
?movieId_embedding/movieId_embedding_weights/strided_slice/stack�
AmovieId_embedding/movieId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
AmovieId_embedding/movieId_embedding_weights/strided_slice/stack_1�
AmovieId_embedding/movieId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
AmovieId_embedding/movieId_embedding_weights/strided_slice/stack_2�
9movieId_embedding/movieId_embedding_weights/strided_sliceStridedSlice:movieId_embedding/movieId_embedding_weights/Shape:output:0HmovieId_embedding/movieId_embedding_weights/strided_slice/stack:output:0JmovieId_embedding/movieId_embedding_weights/strided_slice/stack_1:output:0JmovieId_embedding/movieId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9movieId_embedding/movieId_embedding_weights/strided_slice�
3movieId_embedding/movieId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :25
3movieId_embedding/movieId_embedding_weights/stack/0�
1movieId_embedding/movieId_embedding_weights/stackPack<movieId_embedding/movieId_embedding_weights/stack/0:output:0BmovieId_embedding/movieId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:23
1movieId_embedding/movieId_embedding_weights/stack�
0movieId_embedding/movieId_embedding_weights/TileTile>movieId_embedding/movieId_embedding_weights/Reshape_1:output:0:movieId_embedding/movieId_embedding_weights/stack:output:0*
T0
*0
_output_shapes
:������������������22
0movieId_embedding/movieId_embedding_weights/Tile�
6movieId_embedding/movieId_embedding_weights/zeros_like	ZerosLikeLmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
28
6movieId_embedding/movieId_embedding_weights/zeros_like�
+movieId_embedding/movieId_embedding_weightsSelect9movieId_embedding/movieId_embedding_weights/Tile:output:0:movieId_embedding/movieId_embedding_weights/zeros_like:y:0LmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
2-
+movieId_embedding/movieId_embedding_weights�
2movieId_embedding/movieId_embedding_weights/Cast_1Cast6movieId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:24
2movieId_embedding/movieId_embedding_weights/Cast_1�
9movieId_embedding/movieId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 2;
9movieId_embedding/movieId_embedding_weights/Slice_1/begin�
8movieId_embedding/movieId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:2:
8movieId_embedding/movieId_embedding_weights/Slice_1/size�
3movieId_embedding/movieId_embedding_weights/Slice_1Slice6movieId_embedding/movieId_embedding_weights/Cast_1:y:0BmovieId_embedding/movieId_embedding_weights/Slice_1/begin:output:0AmovieId_embedding/movieId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:25
3movieId_embedding/movieId_embedding_weights/Slice_1�
3movieId_embedding/movieId_embedding_weights/Shape_1Shape4movieId_embedding/movieId_embedding_weights:output:0*
T0*
_output_shapes
:25
3movieId_embedding/movieId_embedding_weights/Shape_1�
9movieId_embedding/movieId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:2;
9movieId_embedding/movieId_embedding_weights/Slice_2/begin�
8movieId_embedding/movieId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
���������2:
8movieId_embedding/movieId_embedding_weights/Slice_2/size�
3movieId_embedding/movieId_embedding_weights/Slice_2Slice<movieId_embedding/movieId_embedding_weights/Shape_1:output:0BmovieId_embedding/movieId_embedding_weights/Slice_2/begin:output:0AmovieId_embedding/movieId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:25
3movieId_embedding/movieId_embedding_weights/Slice_2�
7movieId_embedding/movieId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7movieId_embedding/movieId_embedding_weights/concat/axis�
2movieId_embedding/movieId_embedding_weights/concatConcatV2<movieId_embedding/movieId_embedding_weights/Slice_1:output:0<movieId_embedding/movieId_embedding_weights/Slice_2:output:0@movieId_embedding/movieId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2movieId_embedding/movieId_embedding_weights/concat�
5movieId_embedding/movieId_embedding_weights/Reshape_2Reshape4movieId_embedding/movieId_embedding_weights:output:0;movieId_embedding/movieId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:���������
27
5movieId_embedding/movieId_embedding_weights/Reshape_2�
movieId_embedding/ShapeShape>movieId_embedding/movieId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:2
movieId_embedding/Shape�
%movieId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%movieId_embedding/strided_slice/stack�
'movieId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'movieId_embedding/strided_slice/stack_1�
'movieId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'movieId_embedding/strided_slice/stack_2�
movieId_embedding/strided_sliceStridedSlice movieId_embedding/Shape:output:0.movieId_embedding/strided_slice/stack:output:00movieId_embedding/strided_slice/stack_1:output:00movieId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
movieId_embedding/strided_slice�
!movieId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2#
!movieId_embedding/Reshape/shape/1�
movieId_embedding/Reshape/shapePack(movieId_embedding/strided_slice:output:0*movieId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2!
movieId_embedding/Reshape/shape�
movieId_embedding/ReshapeReshape>movieId_embedding/movieId_embedding_weights/Reshape_2:output:0(movieId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:���������
2
movieId_embedding/Reshapeq
concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat/concat_dim�
concat/concatIdentity"movieId_embedding/Reshape:output:0*
T0*'
_output_shapes
:���������
2
concat/concatj
IdentityIdentityconcat/concat:output:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*5
_input_shapes$
":���������:���������::M I
#
_output_shapes
:���������
"
_user_specified_name
features:MI
#
_output_shapes
:���������
"
_user_specified_name
features
�	
�
-__inference_functional_1_layer_call_fn_213033
inputs_movieid
inputs_userid
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_movieidinputs_useridunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_2125622
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:���������:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
#
_output_shapes
:���������
(
_user_specified_nameinputs/movieId:RN
#
_output_shapes
:���������
'
_user_specified_nameinputs/userId
��
�
!__inference__wrapped_model_211970
movieid

userid{
wfunctional_1_dense_features_movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_211827{
wfunctional_1_dense_features_1_userid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_2119075
1functional_1_dense_matmul_readvariableop_resource6
2functional_1_dense_biasadd_readvariableop_resource7
3functional_1_dense_1_matmul_readvariableop_resource8
4functional_1_dense_1_biasadd_readvariableop_resource7
3functional_1_dense_2_matmul_readvariableop_resource8
4functional_1_dense_2_biasadd_readvariableop_resource
identity��
<functional_1/dense_features/movieId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2>
<functional_1/dense_features/movieId_embedding/ExpandDims/dim�
8functional_1/dense_features/movieId_embedding/ExpandDims
ExpandDimsmovieidEfunctional_1/dense_features/movieId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������2:
8functional_1/dense_features/movieId_embedding/ExpandDims�
Lfunctional_1/dense_features/movieId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
���������2N
Lfunctional_1/dense_features/movieId_embedding/to_sparse_input/ignore_value/x�
Ffunctional_1/dense_features/movieId_embedding/to_sparse_input/NotEqualNotEqualAfunctional_1/dense_features/movieId_embedding/ExpandDims:output:0Ufunctional_1/dense_features/movieId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:���������2H
Ffunctional_1/dense_features/movieId_embedding/to_sparse_input/NotEqual�
Efunctional_1/dense_features/movieId_embedding/to_sparse_input/indicesWhereJfunctional_1/dense_features/movieId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:���������2G
Efunctional_1/dense_features/movieId_embedding/to_sparse_input/indices�
Dfunctional_1/dense_features/movieId_embedding/to_sparse_input/valuesGatherNdAfunctional_1/dense_features/movieId_embedding/ExpandDims:output:0Mfunctional_1/dense_features/movieId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:���������2F
Dfunctional_1/dense_features/movieId_embedding/to_sparse_input/values�
Ifunctional_1/dense_features/movieId_embedding/to_sparse_input/dense_shapeShapeAfunctional_1/dense_features/movieId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2K
Ifunctional_1/dense_features/movieId_embedding/to_sparse_input/dense_shape�
4functional_1/dense_features/movieId_embedding/valuesCastMfunctional_1/dense_features/movieId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:���������26
4functional_1/dense_features/movieId_embedding/values�
Sfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 2U
Sfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice/begin�
Rfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:2T
Rfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice/size�
Mfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/SliceSliceRfunctional_1/dense_features/movieId_embedding/to_sparse_input/dense_shape:output:0\functional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice/begin:output:0[functional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:2O
Mfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice�
Mfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2O
Mfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Const�
Lfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/ProdProdVfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice:output:0Vfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: 2N
Lfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Prod�
Xfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :2Z
Xfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2/indices�
Ufunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2W
Ufunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2/axis�
Pfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2GatherV2Rfunctional_1/dense_features/movieId_embedding/to_sparse_input/dense_shape:output:0afunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2/indices:output:0^functional_1/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 2R
Pfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2�
Nfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Cast/xPackUfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Prod:output:0Yfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:2P
Nfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Cast/x�
Ufunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/SparseReshapeSparseReshapeMfunctional_1/dense_features/movieId_embedding/to_sparse_input/indices:index:0Rfunctional_1/dense_features/movieId_embedding/to_sparse_input/dense_shape:output:0Wfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Cast/x:output:0*-
_output_shapes
:���������:2W
Ufunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/SparseReshape�
^functional_1/dense_features/movieId_embedding/movieId_embedding_weights/SparseReshape/IdentityIdentity8functional_1/dense_features/movieId_embedding/values:y:0*
T0	*#
_output_shapes
:���������2`
^functional_1/dense_features/movieId_embedding/movieId_embedding_weights/SparseReshape/Identity�
Vfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2X
Vfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/GreaterEqual/y�
Tfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/GreaterEqualGreaterEqualgfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0_functional_1/dense_features/movieId_embedding/movieId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:���������2V
Tfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/GreaterEqual�
Mfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/WhereWhereXfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:���������2O
Mfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Where�
Ufunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������2W
Ufunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Reshape/shape�
Ofunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/ReshapeReshapeUfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Where:index:0^functional_1/dense_features/movieId_embedding/movieId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:���������2Q
Ofunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Reshape�
Wfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Y
Wfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1/axis�
Rfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1GatherV2ffunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/SparseReshape:output_indices:0Xfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Reshape:output:0`functional_1/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:���������2T
Rfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1�
Wfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Y
Wfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2/axis�
Rfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2GatherV2gfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0Xfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Reshape:output:0`functional_1/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:���������2T
Rfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2�
Pfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/IdentityIdentitydfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:2R
Pfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Identity�
afunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 2c
afunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/Const�
ofunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows[functional_1/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1:output:0[functional_1/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2:output:0Yfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Identity:output:0jfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:���������:���������:���������:���������2q
ofunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows�
sfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2u
sfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack�
ufunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2w
ufunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1�
ufunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2w
ufunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2�
mfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice�functional_1/dense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0|functional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0~functional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0~functional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2o
mfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice�
ffunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/UniqueUniquefunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:���������:���������2h
ffunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique�
pfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherwfunctional_1_dense_features_movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_211827jfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*�
_class�
~|loc:@functional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/211827*'
_output_shapes
:���������
*
dtype02r
pfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup�
yfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityyfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*�
_class�
~|loc:@functional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/211827*'
_output_shapes
:���������
2{
yfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity�
{functional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1Identity�functional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������
2}
{functional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1�
_functional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparseSparseSegmentMean�functional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0lfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:idx:0vfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:���������
2a
_functional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse�
Wfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2Y
Wfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Reshape_1/shape�
Qfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Reshape_1Reshape�functional_1/dense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0`functional_1/dense_features/movieId_embedding/movieId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:���������2S
Qfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Reshape_1�
Mfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/ShapeShapehfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:2O
Mfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Shape�
[functional_1/dense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2]
[functional_1/dense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack�
]functional_1/dense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2_
]functional_1/dense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_1�
]functional_1/dense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2_
]functional_1/dense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_2�
Ufunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/strided_sliceStridedSliceVfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Shape:output:0dfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack:output:0ffunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_1:output:0ffunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2W
Ufunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/strided_slice�
Ofunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :2Q
Ofunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/stack/0�
Mfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/stackPackXfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/stack/0:output:0^functional_1/dense_features/movieId_embedding/movieId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:2O
Mfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/stack�
Lfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/TileTileZfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Reshape_1:output:0Vfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/stack:output:0*
T0
*0
_output_shapes
:������������������2N
Lfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Tile�
Rfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/zeros_like	ZerosLikehfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
2T
Rfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/zeros_like�
Gfunctional_1/dense_features/movieId_embedding/movieId_embedding_weightsSelectUfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Tile:output:0Vfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/zeros_like:y:0hfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
2I
Gfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights�
Nfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Cast_1CastRfunctional_1/dense_features/movieId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:2P
Nfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Cast_1�
Ufunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 2W
Ufunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice_1/begin�
Tfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:2V
Tfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice_1/size�
Ofunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice_1SliceRfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Cast_1:y:0^functional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice_1/begin:output:0]functional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:2Q
Ofunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice_1�
Ofunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Shape_1ShapePfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights:output:0*
T0*
_output_shapes
:2Q
Ofunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Shape_1�
Ufunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:2W
Ufunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice_2/begin�
Tfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
���������2V
Tfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice_2/size�
Ofunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice_2SliceXfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Shape_1:output:0^functional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice_2/begin:output:0]functional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:2Q
Ofunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice_2�
Sfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Sfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/concat/axis�
Nfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/concatConcatV2Xfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice_1:output:0Xfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Slice_2:output:0\functional_1/dense_features/movieId_embedding/movieId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:2P
Nfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/concat�
Qfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Reshape_2ReshapePfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights:output:0Wfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:���������
2S
Qfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Reshape_2�
3functional_1/dense_features/movieId_embedding/ShapeShapeZfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:25
3functional_1/dense_features/movieId_embedding/Shape�
Afunctional_1/dense_features/movieId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Afunctional_1/dense_features/movieId_embedding/strided_slice/stack�
Cfunctional_1/dense_features/movieId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Cfunctional_1/dense_features/movieId_embedding/strided_slice/stack_1�
Cfunctional_1/dense_features/movieId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Cfunctional_1/dense_features/movieId_embedding/strided_slice/stack_2�
;functional_1/dense_features/movieId_embedding/strided_sliceStridedSlice<functional_1/dense_features/movieId_embedding/Shape:output:0Jfunctional_1/dense_features/movieId_embedding/strided_slice/stack:output:0Lfunctional_1/dense_features/movieId_embedding/strided_slice/stack_1:output:0Lfunctional_1/dense_features/movieId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;functional_1/dense_features/movieId_embedding/strided_slice�
=functional_1/dense_features/movieId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2?
=functional_1/dense_features/movieId_embedding/Reshape/shape/1�
;functional_1/dense_features/movieId_embedding/Reshape/shapePackDfunctional_1/dense_features/movieId_embedding/strided_slice:output:0Ffunctional_1/dense_features/movieId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2=
;functional_1/dense_features/movieId_embedding/Reshape/shape�
5functional_1/dense_features/movieId_embedding/ReshapeReshapeZfunctional_1/dense_features/movieId_embedding/movieId_embedding_weights/Reshape_2:output:0Dfunctional_1/dense_features/movieId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:���������
27
5functional_1/dense_features/movieId_embedding/Reshape�
-functional_1/dense_features/concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2/
-functional_1/dense_features/concat/concat_dim�
)functional_1/dense_features/concat/concatIdentity>functional_1/dense_features/movieId_embedding/Reshape:output:0*
T0*'
_output_shapes
:���������
2+
)functional_1/dense_features/concat/concat�
=functional_1/dense_features_1/userId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2?
=functional_1/dense_features_1/userId_embedding/ExpandDims/dim�
9functional_1/dense_features_1/userId_embedding/ExpandDims
ExpandDimsuseridFfunctional_1/dense_features_1/userId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������2;
9functional_1/dense_features_1/userId_embedding/ExpandDims�
Mfunctional_1/dense_features_1/userId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
���������2O
Mfunctional_1/dense_features_1/userId_embedding/to_sparse_input/ignore_value/x�
Gfunctional_1/dense_features_1/userId_embedding/to_sparse_input/NotEqualNotEqualBfunctional_1/dense_features_1/userId_embedding/ExpandDims:output:0Vfunctional_1/dense_features_1/userId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:���������2I
Gfunctional_1/dense_features_1/userId_embedding/to_sparse_input/NotEqual�
Ffunctional_1/dense_features_1/userId_embedding/to_sparse_input/indicesWhereKfunctional_1/dense_features_1/userId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:���������2H
Ffunctional_1/dense_features_1/userId_embedding/to_sparse_input/indices�
Efunctional_1/dense_features_1/userId_embedding/to_sparse_input/valuesGatherNdBfunctional_1/dense_features_1/userId_embedding/ExpandDims:output:0Nfunctional_1/dense_features_1/userId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:���������2G
Efunctional_1/dense_features_1/userId_embedding/to_sparse_input/values�
Jfunctional_1/dense_features_1/userId_embedding/to_sparse_input/dense_shapeShapeBfunctional_1/dense_features_1/userId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2L
Jfunctional_1/dense_features_1/userId_embedding/to_sparse_input/dense_shape�
5functional_1/dense_features_1/userId_embedding/valuesCastNfunctional_1/dense_features_1/userId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:���������27
5functional_1/dense_features_1/userId_embedding/values�
Sfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 2U
Sfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice/begin�
Rfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:2T
Rfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice/size�
Mfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/SliceSliceSfunctional_1/dense_features_1/userId_embedding/to_sparse_input/dense_shape:output:0\functional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice/begin:output:0[functional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:2O
Mfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice�
Mfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2O
Mfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Const�
Lfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/ProdProdVfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice:output:0Vfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: 2N
Lfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Prod�
Xfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :2Z
Xfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2/indices�
Ufunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2W
Ufunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2/axis�
Pfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2GatherV2Sfunctional_1/dense_features_1/userId_embedding/to_sparse_input/dense_shape:output:0afunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2/indices:output:0^functional_1/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 2R
Pfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2�
Nfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Cast/xPackUfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Prod:output:0Yfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:2P
Nfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Cast/x�
Ufunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/SparseReshapeSparseReshapeNfunctional_1/dense_features_1/userId_embedding/to_sparse_input/indices:index:0Sfunctional_1/dense_features_1/userId_embedding/to_sparse_input/dense_shape:output:0Wfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Cast/x:output:0*-
_output_shapes
:���������:2W
Ufunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/SparseReshape�
^functional_1/dense_features_1/userId_embedding/userId_embedding_weights/SparseReshape/IdentityIdentity9functional_1/dense_features_1/userId_embedding/values:y:0*
T0	*#
_output_shapes
:���������2`
^functional_1/dense_features_1/userId_embedding/userId_embedding_weights/SparseReshape/Identity�
Vfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2X
Vfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/GreaterEqual/y�
Tfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/GreaterEqualGreaterEqualgfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0_functional_1/dense_features_1/userId_embedding/userId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:���������2V
Tfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/GreaterEqual�
Mfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/WhereWhereXfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:���������2O
Mfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Where�
Ufunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������2W
Ufunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Reshape/shape�
Ofunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/ReshapeReshapeUfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Where:index:0^functional_1/dense_features_1/userId_embedding/userId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:���������2Q
Ofunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Reshape�
Wfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Y
Wfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1/axis�
Rfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1GatherV2ffunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/SparseReshape:output_indices:0Xfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Reshape:output:0`functional_1/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:���������2T
Rfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1�
Wfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Y
Wfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2/axis�
Rfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2GatherV2gfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0Xfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Reshape:output:0`functional_1/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:���������2T
Rfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2�
Pfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/IdentityIdentitydfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:2R
Pfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Identity�
afunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 2c
afunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/Const�
ofunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows[functional_1/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1:output:0[functional_1/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2:output:0Yfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Identity:output:0jfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:���������:���������:���������:���������2q
ofunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows�
sfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2u
sfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack�
ufunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2w
ufunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1�
ufunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2w
ufunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2�
mfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice�functional_1/dense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0|functional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0~functional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0~functional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2o
mfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice�
ffunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/UniqueUniquefunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:���������:���������2h
ffunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique�
pfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherwfunctional_1_dense_features_1_userid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_211907jfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*�
_class�
~|loc:@functional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/211907*'
_output_shapes
:���������
*
dtype02r
pfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup�
yfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityyfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*�
_class�
~|loc:@functional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/211907*'
_output_shapes
:���������
2{
yfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity�
{functional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1Identity�functional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������
2}
{functional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1�
_functional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparseSparseSegmentMean�functional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0lfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:idx:0vfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:���������
2a
_functional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse�
Wfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2Y
Wfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Reshape_1/shape�
Qfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Reshape_1Reshape�functional_1/dense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0`functional_1/dense_features_1/userId_embedding/userId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:���������2S
Qfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Reshape_1�
Mfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/ShapeShapehfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:2O
Mfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Shape�
[functional_1/dense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2]
[functional_1/dense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack�
]functional_1/dense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2_
]functional_1/dense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_1�
]functional_1/dense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2_
]functional_1/dense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_2�
Ufunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/strided_sliceStridedSliceVfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Shape:output:0dfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack:output:0ffunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_1:output:0ffunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2W
Ufunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/strided_slice�
Ofunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :2Q
Ofunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/stack/0�
Mfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/stackPackXfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/stack/0:output:0^functional_1/dense_features_1/userId_embedding/userId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:2O
Mfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/stack�
Lfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/TileTileZfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Reshape_1:output:0Vfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/stack:output:0*
T0
*0
_output_shapes
:������������������2N
Lfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Tile�
Rfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/zeros_like	ZerosLikehfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
2T
Rfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/zeros_like�
Gfunctional_1/dense_features_1/userId_embedding/userId_embedding_weightsSelectUfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Tile:output:0Vfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/zeros_like:y:0hfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
2I
Gfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights�
Nfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Cast_1CastSfunctional_1/dense_features_1/userId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:2P
Nfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Cast_1�
Ufunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 2W
Ufunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice_1/begin�
Tfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:2V
Tfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice_1/size�
Ofunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice_1SliceRfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Cast_1:y:0^functional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice_1/begin:output:0]functional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:2Q
Ofunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice_1�
Ofunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Shape_1ShapePfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights:output:0*
T0*
_output_shapes
:2Q
Ofunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Shape_1�
Ufunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:2W
Ufunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice_2/begin�
Tfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
���������2V
Tfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice_2/size�
Ofunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice_2SliceXfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Shape_1:output:0^functional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice_2/begin:output:0]functional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:2Q
Ofunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice_2�
Sfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Sfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/concat/axis�
Nfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/concatConcatV2Xfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice_1:output:0Xfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Slice_2:output:0\functional_1/dense_features_1/userId_embedding/userId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:2P
Nfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/concat�
Qfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Reshape_2ReshapePfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights:output:0Wfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:���������
2S
Qfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Reshape_2�
4functional_1/dense_features_1/userId_embedding/ShapeShapeZfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:26
4functional_1/dense_features_1/userId_embedding/Shape�
Bfunctional_1/dense_features_1/userId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bfunctional_1/dense_features_1/userId_embedding/strided_slice/stack�
Dfunctional_1/dense_features_1/userId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dfunctional_1/dense_features_1/userId_embedding/strided_slice/stack_1�
Dfunctional_1/dense_features_1/userId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dfunctional_1/dense_features_1/userId_embedding/strided_slice/stack_2�
<functional_1/dense_features_1/userId_embedding/strided_sliceStridedSlice=functional_1/dense_features_1/userId_embedding/Shape:output:0Kfunctional_1/dense_features_1/userId_embedding/strided_slice/stack:output:0Mfunctional_1/dense_features_1/userId_embedding/strided_slice/stack_1:output:0Mfunctional_1/dense_features_1/userId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<functional_1/dense_features_1/userId_embedding/strided_slice�
>functional_1/dense_features_1/userId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2@
>functional_1/dense_features_1/userId_embedding/Reshape/shape/1�
<functional_1/dense_features_1/userId_embedding/Reshape/shapePackEfunctional_1/dense_features_1/userId_embedding/strided_slice:output:0Gfunctional_1/dense_features_1/userId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2>
<functional_1/dense_features_1/userId_embedding/Reshape/shape�
6functional_1/dense_features_1/userId_embedding/ReshapeReshapeZfunctional_1/dense_features_1/userId_embedding/userId_embedding_weights/Reshape_2:output:0Efunctional_1/dense_features_1/userId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:���������
28
6functional_1/dense_features_1/userId_embedding/Reshape�
/functional_1/dense_features_1/concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������21
/functional_1/dense_features_1/concat/concat_dim�
+functional_1/dense_features_1/concat/concatIdentity?functional_1/dense_features_1/userId_embedding/Reshape:output:0*
T0*'
_output_shapes
:���������
2-
+functional_1/dense_features_1/concat/concat�
$functional_1/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2&
$functional_1/concatenate/concat/axis�
functional_1/concatenate/concatConcatV22functional_1/dense_features/concat/concat:output:04functional_1/dense_features_1/concat/concat:output:0-functional_1/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2!
functional_1/concatenate/concat�
(functional_1/dense/MatMul/ReadVariableOpReadVariableOp1functional_1_dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02*
(functional_1/dense/MatMul/ReadVariableOp�
functional_1/dense/MatMulMatMul(functional_1/concatenate/concat:output:00functional_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
functional_1/dense/MatMul�
)functional_1/dense/BiasAdd/ReadVariableOpReadVariableOp2functional_1_dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02+
)functional_1/dense/BiasAdd/ReadVariableOp�
functional_1/dense/BiasAddBiasAdd#functional_1/dense/MatMul:product:01functional_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
functional_1/dense/BiasAdd�
functional_1/dense/ReluRelu#functional_1/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
functional_1/dense/Relu�
*functional_1/dense_1/MatMul/ReadVariableOpReadVariableOp3functional_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02,
*functional_1/dense_1/MatMul/ReadVariableOp�
functional_1/dense_1/MatMulMatMul%functional_1/dense/Relu:activations:02functional_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
functional_1/dense_1/MatMul�
+functional_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+functional_1/dense_1/BiasAdd/ReadVariableOp�
functional_1/dense_1/BiasAddBiasAdd%functional_1/dense_1/MatMul:product:03functional_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
functional_1/dense_1/BiasAdd�
functional_1/dense_1/ReluRelu%functional_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
functional_1/dense_1/Relu�
*functional_1/dense_2/MatMul/ReadVariableOpReadVariableOp3functional_1_dense_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02,
*functional_1/dense_2/MatMul/ReadVariableOp�
functional_1/dense_2/MatMulMatMul'functional_1/dense_1/Relu:activations:02functional_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
functional_1/dense_2/MatMul�
+functional_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+functional_1/dense_2/BiasAdd/ReadVariableOp�
functional_1/dense_2/BiasAddBiasAdd%functional_1/dense_2/MatMul:product:03functional_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
functional_1/dense_2/BiasAdd�
functional_1/dense_2/SigmoidSigmoid%functional_1/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
functional_1/dense_2/Sigmoidt
IdentityIdentity functional_1/dense_2/Sigmoid:y:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:���������:���������:::::::::L H
#
_output_shapes
:���������
!
_user_specified_name	movieId:KG
#
_output_shapes
:���������
 
_user_specified_nameuserId
�W
�
__inference__traced_save_213625
file_prefix?
;savev2_dense_features_embedding_weights_read_readvariableopA
=savev2_dense_features_1_embedding_weights_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop/
+savev2_true_negatives_1_read_readvariableop0
,savev2_false_positives_1_read_readvariableop0
,savev2_false_negatives_1_read_readvariableopF
Bsavev2_adam_dense_features_embedding_weights_m_read_readvariableopH
Dsavev2_adam_dense_features_1_embedding_weights_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableopF
Bsavev2_adam_dense_features_embedding_weights_v_read_readvariableopH
Dsavev2_adam_dense_features_1_embedding_weights_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
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
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_e6e60f7743af44a28a7c6f0f785676b3/part2	
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*�
value�B�*BTlayer_with_weights-0/movieId_embedding.Sembedding_weights/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/userId_embedding.Sembedding_weights/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-0/movieId_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBolayer_with_weights-1/userId_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-0/movieId_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBolayer_with_weights-1/userId_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0;savev2_dense_features_embedding_weights_read_readvariableop=savev2_dense_features_1_embedding_weights_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop+savev2_true_positives_1_read_readvariableop+savev2_true_negatives_1_read_readvariableop,savev2_false_positives_1_read_readvariableop,savev2_false_negatives_1_read_readvariableopBsavev2_adam_dense_features_embedding_weights_m_read_readvariableopDsavev2_adam_dense_features_1_embedding_weights_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableopBsavev2_adam_dense_features_embedding_weights_v_read_readvariableopDsavev2_adam_dense_features_1_embedding_weights_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *8
dtypes.
,2*	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :	�
:
��
:
:
:

:
:
:: : : : : : : : : :�:�:�:�:�:�:�:�:	�
:
��
:
:
:

:
:
::	�
:
��
:
:
:

:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�
:&"
 
_output_shapes
:
��
:$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::	
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
: :!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:%!

_output_shapes
:	�
:&"
 
_output_shapes
:
��
:$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$  

_output_shapes

:
: !

_output_shapes
::%"!

_output_shapes
:	�
:&#"
 
_output_shapes
:
��
:$$ 

_output_shapes

:
: %

_output_shapes
:
:$& 

_output_shapes

:

: '

_output_shapes
:
:$( 

_output_shapes

:
: )

_output_shapes
::*

_output_shapes
: 
�
�
L__inference_dense_features_1_layer_call_and_return_conditional_losses_213389
features_movieid
features_userid]
Yuserid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_213349
identity��
userId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
userId_embedding/ExpandDims/dim�
userId_embedding/ExpandDims
ExpandDimsfeatures_userid(userId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������2
userId_embedding/ExpandDims�
/userId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
���������21
/userId_embedding/to_sparse_input/ignore_value/x�
)userId_embedding/to_sparse_input/NotEqualNotEqual$userId_embedding/ExpandDims:output:08userId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:���������2+
)userId_embedding/to_sparse_input/NotEqual�
(userId_embedding/to_sparse_input/indicesWhere-userId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:���������2*
(userId_embedding/to_sparse_input/indices�
'userId_embedding/to_sparse_input/valuesGatherNd$userId_embedding/ExpandDims:output:00userId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:���������2)
'userId_embedding/to_sparse_input/values�
,userId_embedding/to_sparse_input/dense_shapeShape$userId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2.
,userId_embedding/to_sparse_input/dense_shape�
userId_embedding/valuesCast0userId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:���������2
userId_embedding/values�
5userId_embedding/userId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 27
5userId_embedding/userId_embedding_weights/Slice/begin�
4userId_embedding/userId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:26
4userId_embedding/userId_embedding_weights/Slice/size�
/userId_embedding/userId_embedding_weights/SliceSlice5userId_embedding/to_sparse_input/dense_shape:output:0>userId_embedding/userId_embedding_weights/Slice/begin:output:0=userId_embedding/userId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:21
/userId_embedding/userId_embedding_weights/Slice�
/userId_embedding/userId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/userId_embedding/userId_embedding_weights/Const�
.userId_embedding/userId_embedding_weights/ProdProd8userId_embedding/userId_embedding_weights/Slice:output:08userId_embedding/userId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: 20
.userId_embedding/userId_embedding_weights/Prod�
:userId_embedding/userId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :2<
:userId_embedding/userId_embedding_weights/GatherV2/indices�
7userId_embedding/userId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7userId_embedding/userId_embedding_weights/GatherV2/axis�
2userId_embedding/userId_embedding_weights/GatherV2GatherV25userId_embedding/to_sparse_input/dense_shape:output:0CuserId_embedding/userId_embedding_weights/GatherV2/indices:output:0@userId_embedding/userId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 24
2userId_embedding/userId_embedding_weights/GatherV2�
0userId_embedding/userId_embedding_weights/Cast/xPack7userId_embedding/userId_embedding_weights/Prod:output:0;userId_embedding/userId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:22
0userId_embedding/userId_embedding_weights/Cast/x�
7userId_embedding/userId_embedding_weights/SparseReshapeSparseReshape0userId_embedding/to_sparse_input/indices:index:05userId_embedding/to_sparse_input/dense_shape:output:09userId_embedding/userId_embedding_weights/Cast/x:output:0*-
_output_shapes
:���������:29
7userId_embedding/userId_embedding_weights/SparseReshape�
@userId_embedding/userId_embedding_weights/SparseReshape/IdentityIdentityuserId_embedding/values:y:0*
T0	*#
_output_shapes
:���������2B
@userId_embedding/userId_embedding_weights/SparseReshape/Identity�
8userId_embedding/userId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2:
8userId_embedding/userId_embedding_weights/GreaterEqual/y�
6userId_embedding/userId_embedding_weights/GreaterEqualGreaterEqualIuserId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0AuserId_embedding/userId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:���������28
6userId_embedding/userId_embedding_weights/GreaterEqual�
/userId_embedding/userId_embedding_weights/WhereWhere:userId_embedding/userId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:���������21
/userId_embedding/userId_embedding_weights/Where�
7userId_embedding/userId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������29
7userId_embedding/userId_embedding_weights/Reshape/shape�
1userId_embedding/userId_embedding_weights/ReshapeReshape7userId_embedding/userId_embedding_weights/Where:index:0@userId_embedding/userId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:���������23
1userId_embedding/userId_embedding_weights/Reshape�
9userId_embedding/userId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9userId_embedding/userId_embedding_weights/GatherV2_1/axis�
4userId_embedding/userId_embedding_weights/GatherV2_1GatherV2HuserId_embedding/userId_embedding_weights/SparseReshape:output_indices:0:userId_embedding/userId_embedding_weights/Reshape:output:0BuserId_embedding/userId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:���������26
4userId_embedding/userId_embedding_weights/GatherV2_1�
9userId_embedding/userId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9userId_embedding/userId_embedding_weights/GatherV2_2/axis�
4userId_embedding/userId_embedding_weights/GatherV2_2GatherV2IuserId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0:userId_embedding/userId_embedding_weights/Reshape:output:0BuserId_embedding/userId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:���������26
4userId_embedding/userId_embedding_weights/GatherV2_2�
2userId_embedding/userId_embedding_weights/IdentityIdentityFuserId_embedding/userId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:24
2userId_embedding/userId_embedding_weights/Identity�
CuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 2E
CuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/Const�
QuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows=userId_embedding/userId_embedding_weights/GatherV2_1:output:0=userId_embedding/userId_embedding_weights/GatherV2_2:output:0;userId_embedding/userId_embedding_weights/Identity:output:0LuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:���������:���������:���������:���������2S
QuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows�
UuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2W
UuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack�
WuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2Y
WuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1�
WuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2Y
WuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2�
OuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicebuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0^userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0`userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0`userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2Q
OuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice�
HuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/UniqueUniqueauserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:���������:���������2J
HuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique�
RuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherYuserid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_213349LuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*l
_classb
`^loc:@userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/213349*'
_output_shapes
:���������
*
dtype02T
RuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup�
[userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity[userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*l
_classb
`^loc:@userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/213349*'
_output_shapes
:���������
2]
[userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity�
]userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1IdentityduserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������
2_
]userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1�
AuserId_embedding/userId_embedding_weights/embedding_lookup_sparseSparseSegmentMeanfuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0NuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:idx:0XuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:���������
2C
AuserId_embedding/userId_embedding_weights/embedding_lookup_sparse�
9userId_embedding/userId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2;
9userId_embedding/userId_embedding_weights/Reshape_1/shape�
3userId_embedding/userId_embedding_weights/Reshape_1ReshapeguserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0BuserId_embedding/userId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:���������25
3userId_embedding/userId_embedding_weights/Reshape_1�
/userId_embedding/userId_embedding_weights/ShapeShapeJuserId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:21
/userId_embedding/userId_embedding_weights/Shape�
=userId_embedding/userId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
=userId_embedding/userId_embedding_weights/strided_slice/stack�
?userId_embedding/userId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?userId_embedding/userId_embedding_weights/strided_slice/stack_1�
?userId_embedding/userId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?userId_embedding/userId_embedding_weights/strided_slice/stack_2�
7userId_embedding/userId_embedding_weights/strided_sliceStridedSlice8userId_embedding/userId_embedding_weights/Shape:output:0FuserId_embedding/userId_embedding_weights/strided_slice/stack:output:0HuserId_embedding/userId_embedding_weights/strided_slice/stack_1:output:0HuserId_embedding/userId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7userId_embedding/userId_embedding_weights/strided_slice�
1userId_embedding/userId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :23
1userId_embedding/userId_embedding_weights/stack/0�
/userId_embedding/userId_embedding_weights/stackPack:userId_embedding/userId_embedding_weights/stack/0:output:0@userId_embedding/userId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:21
/userId_embedding/userId_embedding_weights/stack�
.userId_embedding/userId_embedding_weights/TileTile<userId_embedding/userId_embedding_weights/Reshape_1:output:08userId_embedding/userId_embedding_weights/stack:output:0*
T0
*0
_output_shapes
:������������������20
.userId_embedding/userId_embedding_weights/Tile�
4userId_embedding/userId_embedding_weights/zeros_like	ZerosLikeJuserId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
26
4userId_embedding/userId_embedding_weights/zeros_like�
)userId_embedding/userId_embedding_weightsSelect7userId_embedding/userId_embedding_weights/Tile:output:08userId_embedding/userId_embedding_weights/zeros_like:y:0JuserId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
2+
)userId_embedding/userId_embedding_weights�
0userId_embedding/userId_embedding_weights/Cast_1Cast5userId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:22
0userId_embedding/userId_embedding_weights/Cast_1�
7userId_embedding/userId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 29
7userId_embedding/userId_embedding_weights/Slice_1/begin�
6userId_embedding/userId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:28
6userId_embedding/userId_embedding_weights/Slice_1/size�
1userId_embedding/userId_embedding_weights/Slice_1Slice4userId_embedding/userId_embedding_weights/Cast_1:y:0@userId_embedding/userId_embedding_weights/Slice_1/begin:output:0?userId_embedding/userId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:23
1userId_embedding/userId_embedding_weights/Slice_1�
1userId_embedding/userId_embedding_weights/Shape_1Shape2userId_embedding/userId_embedding_weights:output:0*
T0*
_output_shapes
:23
1userId_embedding/userId_embedding_weights/Shape_1�
7userId_embedding/userId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:29
7userId_embedding/userId_embedding_weights/Slice_2/begin�
6userId_embedding/userId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
���������28
6userId_embedding/userId_embedding_weights/Slice_2/size�
1userId_embedding/userId_embedding_weights/Slice_2Slice:userId_embedding/userId_embedding_weights/Shape_1:output:0@userId_embedding/userId_embedding_weights/Slice_2/begin:output:0?userId_embedding/userId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:23
1userId_embedding/userId_embedding_weights/Slice_2�
5userId_embedding/userId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5userId_embedding/userId_embedding_weights/concat/axis�
0userId_embedding/userId_embedding_weights/concatConcatV2:userId_embedding/userId_embedding_weights/Slice_1:output:0:userId_embedding/userId_embedding_weights/Slice_2:output:0>userId_embedding/userId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0userId_embedding/userId_embedding_weights/concat�
3userId_embedding/userId_embedding_weights/Reshape_2Reshape2userId_embedding/userId_embedding_weights:output:09userId_embedding/userId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:���������
25
3userId_embedding/userId_embedding_weights/Reshape_2�
userId_embedding/ShapeShape<userId_embedding/userId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:2
userId_embedding/Shape�
$userId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$userId_embedding/strided_slice/stack�
&userId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&userId_embedding/strided_slice/stack_1�
&userId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&userId_embedding/strided_slice/stack_2�
userId_embedding/strided_sliceStridedSliceuserId_embedding/Shape:output:0-userId_embedding/strided_slice/stack:output:0/userId_embedding/strided_slice/stack_1:output:0/userId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
userId_embedding/strided_slice�
 userId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2"
 userId_embedding/Reshape/shape/1�
userId_embedding/Reshape/shapePack'userId_embedding/strided_slice:output:0)userId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
userId_embedding/Reshape/shape�
userId_embedding/ReshapeReshape<userId_embedding/userId_embedding_weights/Reshape_2:output:0'userId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:���������
2
userId_embedding/Reshapeq
concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat/concat_dim
concat/concatIdentity!userId_embedding/Reshape:output:0*
T0*'
_output_shapes
:���������
2
concat/concatj
IdentityIdentityconcat/concat:output:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*5
_input_shapes$
":���������:���������::U Q
#
_output_shapes
:���������
*
_user_specified_namefeatures/movieId:TP
#
_output_shapes
:���������
)
_user_specified_namefeatures/userId
�
�
C__inference_dense_2_layer_call_and_return_conditional_losses_212438

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
:::O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
��
�
H__inference_functional_1_layer_call_and_return_conditional_losses_212989
inputs_movieid
inputs_useridn
jdense_features_movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_212846n
jdense_features_1_userid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_212926(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity��
/dense_features/movieId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������21
/dense_features/movieId_embedding/ExpandDims/dim�
+dense_features/movieId_embedding/ExpandDims
ExpandDimsinputs_movieid8dense_features/movieId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������2-
+dense_features/movieId_embedding/ExpandDims�
?dense_features/movieId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
���������2A
?dense_features/movieId_embedding/to_sparse_input/ignore_value/x�
9dense_features/movieId_embedding/to_sparse_input/NotEqualNotEqual4dense_features/movieId_embedding/ExpandDims:output:0Hdense_features/movieId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:���������2;
9dense_features/movieId_embedding/to_sparse_input/NotEqual�
8dense_features/movieId_embedding/to_sparse_input/indicesWhere=dense_features/movieId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:���������2:
8dense_features/movieId_embedding/to_sparse_input/indices�
7dense_features/movieId_embedding/to_sparse_input/valuesGatherNd4dense_features/movieId_embedding/ExpandDims:output:0@dense_features/movieId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:���������29
7dense_features/movieId_embedding/to_sparse_input/values�
<dense_features/movieId_embedding/to_sparse_input/dense_shapeShape4dense_features/movieId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2>
<dense_features/movieId_embedding/to_sparse_input/dense_shape�
'dense_features/movieId_embedding/valuesCast@dense_features/movieId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:���������2)
'dense_features/movieId_embedding/values�
Fdense_features/movieId_embedding/movieId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 2H
Fdense_features/movieId_embedding/movieId_embedding_weights/Slice/begin�
Edense_features/movieId_embedding/movieId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:2G
Edense_features/movieId_embedding/movieId_embedding_weights/Slice/size�
@dense_features/movieId_embedding/movieId_embedding_weights/SliceSliceEdense_features/movieId_embedding/to_sparse_input/dense_shape:output:0Odense_features/movieId_embedding/movieId_embedding_weights/Slice/begin:output:0Ndense_features/movieId_embedding/movieId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:2B
@dense_features/movieId_embedding/movieId_embedding_weights/Slice�
@dense_features/movieId_embedding/movieId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2B
@dense_features/movieId_embedding/movieId_embedding_weights/Const�
?dense_features/movieId_embedding/movieId_embedding_weights/ProdProdIdense_features/movieId_embedding/movieId_embedding_weights/Slice:output:0Idense_features/movieId_embedding/movieId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: 2A
?dense_features/movieId_embedding/movieId_embedding_weights/Prod�
Kdense_features/movieId_embedding/movieId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :2M
Kdense_features/movieId_embedding/movieId_embedding_weights/GatherV2/indices�
Hdense_features/movieId_embedding/movieId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hdense_features/movieId_embedding/movieId_embedding_weights/GatherV2/axis�
Cdense_features/movieId_embedding/movieId_embedding_weights/GatherV2GatherV2Edense_features/movieId_embedding/to_sparse_input/dense_shape:output:0Tdense_features/movieId_embedding/movieId_embedding_weights/GatherV2/indices:output:0Qdense_features/movieId_embedding/movieId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 2E
Cdense_features/movieId_embedding/movieId_embedding_weights/GatherV2�
Adense_features/movieId_embedding/movieId_embedding_weights/Cast/xPackHdense_features/movieId_embedding/movieId_embedding_weights/Prod:output:0Ldense_features/movieId_embedding/movieId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:2C
Adense_features/movieId_embedding/movieId_embedding_weights/Cast/x�
Hdense_features/movieId_embedding/movieId_embedding_weights/SparseReshapeSparseReshape@dense_features/movieId_embedding/to_sparse_input/indices:index:0Edense_features/movieId_embedding/to_sparse_input/dense_shape:output:0Jdense_features/movieId_embedding/movieId_embedding_weights/Cast/x:output:0*-
_output_shapes
:���������:2J
Hdense_features/movieId_embedding/movieId_embedding_weights/SparseReshape�
Qdense_features/movieId_embedding/movieId_embedding_weights/SparseReshape/IdentityIdentity+dense_features/movieId_embedding/values:y:0*
T0	*#
_output_shapes
:���������2S
Qdense_features/movieId_embedding/movieId_embedding_weights/SparseReshape/Identity�
Idense_features/movieId_embedding/movieId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2K
Idense_features/movieId_embedding/movieId_embedding_weights/GreaterEqual/y�
Gdense_features/movieId_embedding/movieId_embedding_weights/GreaterEqualGreaterEqualZdense_features/movieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0Rdense_features/movieId_embedding/movieId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:���������2I
Gdense_features/movieId_embedding/movieId_embedding_weights/GreaterEqual�
@dense_features/movieId_embedding/movieId_embedding_weights/WhereWhereKdense_features/movieId_embedding/movieId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:���������2B
@dense_features/movieId_embedding/movieId_embedding_weights/Where�
Hdense_features/movieId_embedding/movieId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������2J
Hdense_features/movieId_embedding/movieId_embedding_weights/Reshape/shape�
Bdense_features/movieId_embedding/movieId_embedding_weights/ReshapeReshapeHdense_features/movieId_embedding/movieId_embedding_weights/Where:index:0Qdense_features/movieId_embedding/movieId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:���������2D
Bdense_features/movieId_embedding/movieId_embedding_weights/Reshape�
Jdense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jdense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1/axis�
Edense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1GatherV2Ydense_features/movieId_embedding/movieId_embedding_weights/SparseReshape:output_indices:0Kdense_features/movieId_embedding/movieId_embedding_weights/Reshape:output:0Sdense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:���������2G
Edense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1�
Jdense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jdense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2/axis�
Edense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2GatherV2Zdense_features/movieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0Kdense_features/movieId_embedding/movieId_embedding_weights/Reshape:output:0Sdense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:���������2G
Edense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2�
Cdense_features/movieId_embedding/movieId_embedding_weights/IdentityIdentityWdense_features/movieId_embedding/movieId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:2E
Cdense_features/movieId_embedding/movieId_embedding_weights/Identity�
Tdense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 2V
Tdense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/Const�
bdense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsNdense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1:output:0Ndense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2:output:0Ldense_features/movieId_embedding/movieId_embedding_weights/Identity:output:0]dense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:���������:���������:���������:���������2d
bdense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows�
fdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2h
fdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack�
hdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2j
hdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1�
hdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2j
hdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2�
`dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicesdense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0odense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0qdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0qdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2b
`dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice�
Ydense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/UniqueUniquerdense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:���������:���������2[
Ydense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique�
cdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherjdense_features_movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_212846]dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*}
_classs
qoloc:@dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/212846*'
_output_shapes
:���������
*
dtype02e
cdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup�
ldense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityldense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*}
_classs
qoloc:@dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/212846*'
_output_shapes
:���������
2n
ldense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity�
ndense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1Identityudense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������
2p
ndense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1�
Rdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparseSparseSegmentMeanwdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0_dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:idx:0idense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:���������
2T
Rdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse�
Jdense_features/movieId_embedding/movieId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2L
Jdense_features/movieId_embedding/movieId_embedding_weights/Reshape_1/shape�
Ddense_features/movieId_embedding/movieId_embedding_weights/Reshape_1Reshapexdense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0Sdense_features/movieId_embedding/movieId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:���������2F
Ddense_features/movieId_embedding/movieId_embedding_weights/Reshape_1�
@dense_features/movieId_embedding/movieId_embedding_weights/ShapeShape[dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:2B
@dense_features/movieId_embedding/movieId_embedding_weights/Shape�
Ndense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2P
Ndense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack�
Pdense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2R
Pdense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_1�
Pdense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2R
Pdense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_2�
Hdense_features/movieId_embedding/movieId_embedding_weights/strided_sliceStridedSliceIdense_features/movieId_embedding/movieId_embedding_weights/Shape:output:0Wdense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack:output:0Ydense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_1:output:0Ydense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2J
Hdense_features/movieId_embedding/movieId_embedding_weights/strided_slice�
Bdense_features/movieId_embedding/movieId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :2D
Bdense_features/movieId_embedding/movieId_embedding_weights/stack/0�
@dense_features/movieId_embedding/movieId_embedding_weights/stackPackKdense_features/movieId_embedding/movieId_embedding_weights/stack/0:output:0Qdense_features/movieId_embedding/movieId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:2B
@dense_features/movieId_embedding/movieId_embedding_weights/stack�
?dense_features/movieId_embedding/movieId_embedding_weights/TileTileMdense_features/movieId_embedding/movieId_embedding_weights/Reshape_1:output:0Idense_features/movieId_embedding/movieId_embedding_weights/stack:output:0*
T0
*0
_output_shapes
:������������������2A
?dense_features/movieId_embedding/movieId_embedding_weights/Tile�
Edense_features/movieId_embedding/movieId_embedding_weights/zeros_like	ZerosLike[dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
2G
Edense_features/movieId_embedding/movieId_embedding_weights/zeros_like�
:dense_features/movieId_embedding/movieId_embedding_weightsSelectHdense_features/movieId_embedding/movieId_embedding_weights/Tile:output:0Idense_features/movieId_embedding/movieId_embedding_weights/zeros_like:y:0[dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
2<
:dense_features/movieId_embedding/movieId_embedding_weights�
Adense_features/movieId_embedding/movieId_embedding_weights/Cast_1CastEdense_features/movieId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:2C
Adense_features/movieId_embedding/movieId_embedding_weights/Cast_1�
Hdense_features/movieId_embedding/movieId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 2J
Hdense_features/movieId_embedding/movieId_embedding_weights/Slice_1/begin�
Gdense_features/movieId_embedding/movieId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:2I
Gdense_features/movieId_embedding/movieId_embedding_weights/Slice_1/size�
Bdense_features/movieId_embedding/movieId_embedding_weights/Slice_1SliceEdense_features/movieId_embedding/movieId_embedding_weights/Cast_1:y:0Qdense_features/movieId_embedding/movieId_embedding_weights/Slice_1/begin:output:0Pdense_features/movieId_embedding/movieId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:2D
Bdense_features/movieId_embedding/movieId_embedding_weights/Slice_1�
Bdense_features/movieId_embedding/movieId_embedding_weights/Shape_1ShapeCdense_features/movieId_embedding/movieId_embedding_weights:output:0*
T0*
_output_shapes
:2D
Bdense_features/movieId_embedding/movieId_embedding_weights/Shape_1�
Hdense_features/movieId_embedding/movieId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:2J
Hdense_features/movieId_embedding/movieId_embedding_weights/Slice_2/begin�
Gdense_features/movieId_embedding/movieId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
���������2I
Gdense_features/movieId_embedding/movieId_embedding_weights/Slice_2/size�
Bdense_features/movieId_embedding/movieId_embedding_weights/Slice_2SliceKdense_features/movieId_embedding/movieId_embedding_weights/Shape_1:output:0Qdense_features/movieId_embedding/movieId_embedding_weights/Slice_2/begin:output:0Pdense_features/movieId_embedding/movieId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:2D
Bdense_features/movieId_embedding/movieId_embedding_weights/Slice_2�
Fdense_features/movieId_embedding/movieId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fdense_features/movieId_embedding/movieId_embedding_weights/concat/axis�
Adense_features/movieId_embedding/movieId_embedding_weights/concatConcatV2Kdense_features/movieId_embedding/movieId_embedding_weights/Slice_1:output:0Kdense_features/movieId_embedding/movieId_embedding_weights/Slice_2:output:0Odense_features/movieId_embedding/movieId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:2C
Adense_features/movieId_embedding/movieId_embedding_weights/concat�
Ddense_features/movieId_embedding/movieId_embedding_weights/Reshape_2ReshapeCdense_features/movieId_embedding/movieId_embedding_weights:output:0Jdense_features/movieId_embedding/movieId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:���������
2F
Ddense_features/movieId_embedding/movieId_embedding_weights/Reshape_2�
&dense_features/movieId_embedding/ShapeShapeMdense_features/movieId_embedding/movieId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:2(
&dense_features/movieId_embedding/Shape�
4dense_features/movieId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4dense_features/movieId_embedding/strided_slice/stack�
6dense_features/movieId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6dense_features/movieId_embedding/strided_slice/stack_1�
6dense_features/movieId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6dense_features/movieId_embedding/strided_slice/stack_2�
.dense_features/movieId_embedding/strided_sliceStridedSlice/dense_features/movieId_embedding/Shape:output:0=dense_features/movieId_embedding/strided_slice/stack:output:0?dense_features/movieId_embedding/strided_slice/stack_1:output:0?dense_features/movieId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.dense_features/movieId_embedding/strided_slice�
0dense_features/movieId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
22
0dense_features/movieId_embedding/Reshape/shape/1�
.dense_features/movieId_embedding/Reshape/shapePack7dense_features/movieId_embedding/strided_slice:output:09dense_features/movieId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:20
.dense_features/movieId_embedding/Reshape/shape�
(dense_features/movieId_embedding/ReshapeReshapeMdense_features/movieId_embedding/movieId_embedding_weights/Reshape_2:output:07dense_features/movieId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:���������
2*
(dense_features/movieId_embedding/Reshape�
 dense_features/concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 dense_features/concat/concat_dim�
dense_features/concat/concatIdentity1dense_features/movieId_embedding/Reshape:output:0*
T0*'
_output_shapes
:���������
2
dense_features/concat/concat�
0dense_features_1/userId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������22
0dense_features_1/userId_embedding/ExpandDims/dim�
,dense_features_1/userId_embedding/ExpandDims
ExpandDimsinputs_userid9dense_features_1/userId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������2.
,dense_features_1/userId_embedding/ExpandDims�
@dense_features_1/userId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
���������2B
@dense_features_1/userId_embedding/to_sparse_input/ignore_value/x�
:dense_features_1/userId_embedding/to_sparse_input/NotEqualNotEqual5dense_features_1/userId_embedding/ExpandDims:output:0Idense_features_1/userId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:���������2<
:dense_features_1/userId_embedding/to_sparse_input/NotEqual�
9dense_features_1/userId_embedding/to_sparse_input/indicesWhere>dense_features_1/userId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:���������2;
9dense_features_1/userId_embedding/to_sparse_input/indices�
8dense_features_1/userId_embedding/to_sparse_input/valuesGatherNd5dense_features_1/userId_embedding/ExpandDims:output:0Adense_features_1/userId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:���������2:
8dense_features_1/userId_embedding/to_sparse_input/values�
=dense_features_1/userId_embedding/to_sparse_input/dense_shapeShape5dense_features_1/userId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2?
=dense_features_1/userId_embedding/to_sparse_input/dense_shape�
(dense_features_1/userId_embedding/valuesCastAdense_features_1/userId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:���������2*
(dense_features_1/userId_embedding/values�
Fdense_features_1/userId_embedding/userId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 2H
Fdense_features_1/userId_embedding/userId_embedding_weights/Slice/begin�
Edense_features_1/userId_embedding/userId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:2G
Edense_features_1/userId_embedding/userId_embedding_weights/Slice/size�
@dense_features_1/userId_embedding/userId_embedding_weights/SliceSliceFdense_features_1/userId_embedding/to_sparse_input/dense_shape:output:0Odense_features_1/userId_embedding/userId_embedding_weights/Slice/begin:output:0Ndense_features_1/userId_embedding/userId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:2B
@dense_features_1/userId_embedding/userId_embedding_weights/Slice�
@dense_features_1/userId_embedding/userId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2B
@dense_features_1/userId_embedding/userId_embedding_weights/Const�
?dense_features_1/userId_embedding/userId_embedding_weights/ProdProdIdense_features_1/userId_embedding/userId_embedding_weights/Slice:output:0Idense_features_1/userId_embedding/userId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: 2A
?dense_features_1/userId_embedding/userId_embedding_weights/Prod�
Kdense_features_1/userId_embedding/userId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :2M
Kdense_features_1/userId_embedding/userId_embedding_weights/GatherV2/indices�
Hdense_features_1/userId_embedding/userId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hdense_features_1/userId_embedding/userId_embedding_weights/GatherV2/axis�
Cdense_features_1/userId_embedding/userId_embedding_weights/GatherV2GatherV2Fdense_features_1/userId_embedding/to_sparse_input/dense_shape:output:0Tdense_features_1/userId_embedding/userId_embedding_weights/GatherV2/indices:output:0Qdense_features_1/userId_embedding/userId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 2E
Cdense_features_1/userId_embedding/userId_embedding_weights/GatherV2�
Adense_features_1/userId_embedding/userId_embedding_weights/Cast/xPackHdense_features_1/userId_embedding/userId_embedding_weights/Prod:output:0Ldense_features_1/userId_embedding/userId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:2C
Adense_features_1/userId_embedding/userId_embedding_weights/Cast/x�
Hdense_features_1/userId_embedding/userId_embedding_weights/SparseReshapeSparseReshapeAdense_features_1/userId_embedding/to_sparse_input/indices:index:0Fdense_features_1/userId_embedding/to_sparse_input/dense_shape:output:0Jdense_features_1/userId_embedding/userId_embedding_weights/Cast/x:output:0*-
_output_shapes
:���������:2J
Hdense_features_1/userId_embedding/userId_embedding_weights/SparseReshape�
Qdense_features_1/userId_embedding/userId_embedding_weights/SparseReshape/IdentityIdentity,dense_features_1/userId_embedding/values:y:0*
T0	*#
_output_shapes
:���������2S
Qdense_features_1/userId_embedding/userId_embedding_weights/SparseReshape/Identity�
Idense_features_1/userId_embedding/userId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2K
Idense_features_1/userId_embedding/userId_embedding_weights/GreaterEqual/y�
Gdense_features_1/userId_embedding/userId_embedding_weights/GreaterEqualGreaterEqualZdense_features_1/userId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0Rdense_features_1/userId_embedding/userId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:���������2I
Gdense_features_1/userId_embedding/userId_embedding_weights/GreaterEqual�
@dense_features_1/userId_embedding/userId_embedding_weights/WhereWhereKdense_features_1/userId_embedding/userId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:���������2B
@dense_features_1/userId_embedding/userId_embedding_weights/Where�
Hdense_features_1/userId_embedding/userId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������2J
Hdense_features_1/userId_embedding/userId_embedding_weights/Reshape/shape�
Bdense_features_1/userId_embedding/userId_embedding_weights/ReshapeReshapeHdense_features_1/userId_embedding/userId_embedding_weights/Where:index:0Qdense_features_1/userId_embedding/userId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:���������2D
Bdense_features_1/userId_embedding/userId_embedding_weights/Reshape�
Jdense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jdense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1/axis�
Edense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1GatherV2Ydense_features_1/userId_embedding/userId_embedding_weights/SparseReshape:output_indices:0Kdense_features_1/userId_embedding/userId_embedding_weights/Reshape:output:0Sdense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:���������2G
Edense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1�
Jdense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jdense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2/axis�
Edense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2GatherV2Zdense_features_1/userId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0Kdense_features_1/userId_embedding/userId_embedding_weights/Reshape:output:0Sdense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:���������2G
Edense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2�
Cdense_features_1/userId_embedding/userId_embedding_weights/IdentityIdentityWdense_features_1/userId_embedding/userId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:2E
Cdense_features_1/userId_embedding/userId_embedding_weights/Identity�
Tdense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 2V
Tdense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/Const�
bdense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsNdense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1:output:0Ndense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2:output:0Ldense_features_1/userId_embedding/userId_embedding_weights/Identity:output:0]dense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:���������:���������:���������:���������2d
bdense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows�
fdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2h
fdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack�
hdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2j
hdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1�
hdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2j
hdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2�
`dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicesdense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0odense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0qdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0qdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2b
`dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice�
Ydense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/UniqueUniquerdense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:���������:���������2[
Ydense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique�
cdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherjdense_features_1_userid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_212926]dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*}
_classs
qoloc:@dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/212926*'
_output_shapes
:���������
*
dtype02e
cdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup�
ldense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityldense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*}
_classs
qoloc:@dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/212926*'
_output_shapes
:���������
2n
ldense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity�
ndense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1Identityudense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������
2p
ndense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1�
Rdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparseSparseSegmentMeanwdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0_dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:idx:0idense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:���������
2T
Rdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse�
Jdense_features_1/userId_embedding/userId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2L
Jdense_features_1/userId_embedding/userId_embedding_weights/Reshape_1/shape�
Ddense_features_1/userId_embedding/userId_embedding_weights/Reshape_1Reshapexdense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0Sdense_features_1/userId_embedding/userId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:���������2F
Ddense_features_1/userId_embedding/userId_embedding_weights/Reshape_1�
@dense_features_1/userId_embedding/userId_embedding_weights/ShapeShape[dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:2B
@dense_features_1/userId_embedding/userId_embedding_weights/Shape�
Ndense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2P
Ndense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack�
Pdense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2R
Pdense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_1�
Pdense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2R
Pdense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_2�
Hdense_features_1/userId_embedding/userId_embedding_weights/strided_sliceStridedSliceIdense_features_1/userId_embedding/userId_embedding_weights/Shape:output:0Wdense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack:output:0Ydense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_1:output:0Ydense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2J
Hdense_features_1/userId_embedding/userId_embedding_weights/strided_slice�
Bdense_features_1/userId_embedding/userId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :2D
Bdense_features_1/userId_embedding/userId_embedding_weights/stack/0�
@dense_features_1/userId_embedding/userId_embedding_weights/stackPackKdense_features_1/userId_embedding/userId_embedding_weights/stack/0:output:0Qdense_features_1/userId_embedding/userId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:2B
@dense_features_1/userId_embedding/userId_embedding_weights/stack�
?dense_features_1/userId_embedding/userId_embedding_weights/TileTileMdense_features_1/userId_embedding/userId_embedding_weights/Reshape_1:output:0Idense_features_1/userId_embedding/userId_embedding_weights/stack:output:0*
T0
*0
_output_shapes
:������������������2A
?dense_features_1/userId_embedding/userId_embedding_weights/Tile�
Edense_features_1/userId_embedding/userId_embedding_weights/zeros_like	ZerosLike[dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
2G
Edense_features_1/userId_embedding/userId_embedding_weights/zeros_like�
:dense_features_1/userId_embedding/userId_embedding_weightsSelectHdense_features_1/userId_embedding/userId_embedding_weights/Tile:output:0Idense_features_1/userId_embedding/userId_embedding_weights/zeros_like:y:0[dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:���������
2<
:dense_features_1/userId_embedding/userId_embedding_weights�
Adense_features_1/userId_embedding/userId_embedding_weights/Cast_1CastFdense_features_1/userId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:2C
Adense_features_1/userId_embedding/userId_embedding_weights/Cast_1�
Hdense_features_1/userId_embedding/userId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 2J
Hdense_features_1/userId_embedding/userId_embedding_weights/Slice_1/begin�
Gdense_features_1/userId_embedding/userId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:2I
Gdense_features_1/userId_embedding/userId_embedding_weights/Slice_1/size�
Bdense_features_1/userId_embedding/userId_embedding_weights/Slice_1SliceEdense_features_1/userId_embedding/userId_embedding_weights/Cast_1:y:0Qdense_features_1/userId_embedding/userId_embedding_weights/Slice_1/begin:output:0Pdense_features_1/userId_embedding/userId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:2D
Bdense_features_1/userId_embedding/userId_embedding_weights/Slice_1�
Bdense_features_1/userId_embedding/userId_embedding_weights/Shape_1ShapeCdense_features_1/userId_embedding/userId_embedding_weights:output:0*
T0*
_output_shapes
:2D
Bdense_features_1/userId_embedding/userId_embedding_weights/Shape_1�
Hdense_features_1/userId_embedding/userId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:2J
Hdense_features_1/userId_embedding/userId_embedding_weights/Slice_2/begin�
Gdense_features_1/userId_embedding/userId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
���������2I
Gdense_features_1/userId_embedding/userId_embedding_weights/Slice_2/size�
Bdense_features_1/userId_embedding/userId_embedding_weights/Slice_2SliceKdense_features_1/userId_embedding/userId_embedding_weights/Shape_1:output:0Qdense_features_1/userId_embedding/userId_embedding_weights/Slice_2/begin:output:0Pdense_features_1/userId_embedding/userId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:2D
Bdense_features_1/userId_embedding/userId_embedding_weights/Slice_2�
Fdense_features_1/userId_embedding/userId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fdense_features_1/userId_embedding/userId_embedding_weights/concat/axis�
Adense_features_1/userId_embedding/userId_embedding_weights/concatConcatV2Kdense_features_1/userId_embedding/userId_embedding_weights/Slice_1:output:0Kdense_features_1/userId_embedding/userId_embedding_weights/Slice_2:output:0Odense_features_1/userId_embedding/userId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:2C
Adense_features_1/userId_embedding/userId_embedding_weights/concat�
Ddense_features_1/userId_embedding/userId_embedding_weights/Reshape_2ReshapeCdense_features_1/userId_embedding/userId_embedding_weights:output:0Jdense_features_1/userId_embedding/userId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:���������
2F
Ddense_features_1/userId_embedding/userId_embedding_weights/Reshape_2�
'dense_features_1/userId_embedding/ShapeShapeMdense_features_1/userId_embedding/userId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:2)
'dense_features_1/userId_embedding/Shape�
5dense_features_1/userId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5dense_features_1/userId_embedding/strided_slice/stack�
7dense_features_1/userId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/userId_embedding/strided_slice/stack_1�
7dense_features_1/userId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/userId_embedding/strided_slice/stack_2�
/dense_features_1/userId_embedding/strided_sliceStridedSlice0dense_features_1/userId_embedding/Shape:output:0>dense_features_1/userId_embedding/strided_slice/stack:output:0@dense_features_1/userId_embedding/strided_slice/stack_1:output:0@dense_features_1/userId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/dense_features_1/userId_embedding/strided_slice�
1dense_features_1/userId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
23
1dense_features_1/userId_embedding/Reshape/shape/1�
/dense_features_1/userId_embedding/Reshape/shapePack8dense_features_1/userId_embedding/strided_slice:output:0:dense_features_1/userId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:21
/dense_features_1/userId_embedding/Reshape/shape�
)dense_features_1/userId_embedding/ReshapeReshapeMdense_features_1/userId_embedding/userId_embedding_weights/Reshape_2:output:08dense_features_1/userId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:���������
2+
)dense_features_1/userId_embedding/Reshape�
"dense_features_1/concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"dense_features_1/concat/concat_dim�
dense_features_1/concat/concatIdentity2dense_features_1/userId_embedding/Reshape:output:0*
T0*'
_output_shapes
:���������
2 
dense_features_1/concat/concatt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis�
concatenate/concatConcatV2%dense_features/concat/concat:output:0'dense_features_1/concat/concat:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatenate/concat�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2

dense/Relu�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
dense_1/Relu�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_2/Sigmoidg
IdentityIdentitydense_2/Sigmoid:y:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:���������:���������:::::::::S O
#
_output_shapes
:���������
(
_user_specified_nameinputs/movieId:RN
#
_output_shapes
:���������
'
_user_specified_nameinputs/userId
�
�
H__inference_functional_1_layer_call_and_return_conditional_losses_212513

inputs
inputs_1
dense_features_212490
dense_features_1_212493
dense_212497
dense_212499
dense_1_212502
dense_1_212504
dense_2_212507
dense_2_212509
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�&dense_features/StatefulPartitionedCall�(dense_features_1/StatefulPartitionedCall�
&dense_features/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1dense_features_212490*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_2120602(
&dense_features/StatefulPartitionedCall�
(dense_features_1/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1dense_features_1_212493*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_dense_features_1_layer_call_and_return_conditional_losses_2122512*
(dense_features_1/StatefulPartitionedCall�
concatenate/PartitionedCallPartitionedCall/dense_features/StatefulPartitionedCall:output:01dense_features_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_2123642
concatenate/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_212497dense_212499*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2123842
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_212502dense_1_212504*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2124112!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_212507dense_2_212509*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2124382!
dense_2/StatefulPartitionedCall�
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall'^dense_features/StatefulPartitionedCall)^dense_features_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:���������:���������::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2P
&dense_features/StatefulPartitionedCall&dense_features/StatefulPartitionedCall2T
(dense_features_1/StatefulPartitionedCall(dense_features_1/StatefulPartitionedCall:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_dense_1_layer_call_and_return_conditional_losses_212411

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������
2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
:::O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�	
�
-__inference_functional_1_layer_call_fn_213011
inputs_movieid
inputs_userid
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_movieidinputs_useridunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_2125132
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:���������:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
#
_output_shapes
:���������
(
_user_specified_nameinputs/movieId:RN
#
_output_shapes
:���������
'
_user_specified_nameinputs/userId"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
7
movieId,
serving_default_movieId:0���������
5
userId+
serving_default_userId:0���������;
dense_20
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
��
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
layer_with_weights-4
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"��
_tf_keras_networkރ{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "movieId"}, "name": "movieId", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "userId"}, "name": "userId", "inbound_nodes": []}, {"class_name": "DenseFeatures", "config": {"name": "dense_features", "trainable": true, "dtype": "float32", "feature_columns": [{"class_name": "EmbeddingColumn", "config": {"categorical_column": {"class_name": "IdentityCategoricalColumn", "config": {"key": "movieId", "number_buckets": 1001, "default_value": null}}, "dimension": 10, "combiner": "mean", "initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.31622776601683794, "seed": null, "dtype": "float32"}}, "ckpt_to_load_from": null, "tensor_name_in_ckpt": null, "max_norm": null, "trainable": true, "use_safe_embedding_lookup": true}}], "partitioner": null}, "name": "dense_features", "inbound_nodes": [{"movieId": ["movieId", 0, 0, {}], "userId": ["userId", 0, 0, {}]}]}, {"class_name": "DenseFeatures", "config": {"name": "dense_features_1", "trainable": true, "dtype": "float32", "feature_columns": [{"class_name": "EmbeddingColumn", "config": {"categorical_column": {"class_name": "IdentityCategoricalColumn", "config": {"key": "userId", "number_buckets": 30001, "default_value": null}}, "dimension": 10, "combiner": "mean", "initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.31622776601683794, "seed": null, "dtype": "float32"}}, "ckpt_to_load_from": null, "tensor_name_in_ckpt": null, "max_norm": null, "trainable": true, "use_safe_embedding_lookup": true}}], "partitioner": null}, "name": "dense_features_1", "inbound_nodes": [{"movieId": ["movieId", 0, 0, {}], "userId": ["userId", 0, 0, {}]}]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["dense_features", 0, 0, {}], ["dense_features_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": {"movieId": ["movieId", 0, 0], "userId": ["userId", 0, 0]}, "output_layers": [["dense_2", 0, 0]]}, "build_input_shape": {"movieId": {"class_name": "TensorShape", "items": [null]}, "userId": {"class_name": "TensorShape", "items": [null]}}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "movieId"}, "name": "movieId", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "userId"}, "name": "userId", "inbound_nodes": []}, {"class_name": "DenseFeatures", "config": {"name": "dense_features", "trainable": true, "dtype": "float32", "feature_columns": [{"class_name": "EmbeddingColumn", "config": {"categorical_column": {"class_name": "IdentityCategoricalColumn", "config": {"key": "movieId", "number_buckets": 1001, "default_value": null}}, "dimension": 10, "combiner": "mean", "initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.31622776601683794, "seed": null, "dtype": "float32"}}, "ckpt_to_load_from": null, "tensor_name_in_ckpt": null, "max_norm": null, "trainable": true, "use_safe_embedding_lookup": true}}], "partitioner": null}, "name": "dense_features", "inbound_nodes": [{"movieId": ["movieId", 0, 0, {}], "userId": ["userId", 0, 0, {}]}]}, {"class_name": "DenseFeatures", "config": {"name": "dense_features_1", "trainable": true, "dtype": "float32", "feature_columns": [{"class_name": "EmbeddingColumn", "config": {"categorical_column": {"class_name": "IdentityCategoricalColumn", "config": {"key": "userId", "number_buckets": 30001, "default_value": null}}, "dimension": 10, "combiner": "mean", "initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.31622776601683794, "seed": null, "dtype": "float32"}}, "ckpt_to_load_from": null, "tensor_name_in_ckpt": null, "max_norm": null, "trainable": true, "use_safe_embedding_lookup": true}}], "partitioner": null}, "name": "dense_features_1", "inbound_nodes": [{"movieId": ["movieId", 0, 0, {}], "userId": ["userId", 0, 0, {}]}]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["dense_features", 0, 0, {}], ["dense_features_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": {"movieId": ["movieId", 0, 0], "userId": ["userId", 0, 0]}, "output_layers": [["dense_2", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy", {"class_name": "AUC", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}, {"class_name": "AUC", "config": {"name": "auc_1", "dtype": "float32", "num_thresholds": 200, "curve": "PR", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "movieId", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "movieId"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "userId", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "userId"}}
�	
_feature_columns

_resources
'#movieId_embedding/embedding_weights
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "DenseFeatures", "name": "dense_features", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_features", "trainable": true, "dtype": "float32", "feature_columns": [{"class_name": "EmbeddingColumn", "config": {"categorical_column": {"class_name": "IdentityCategoricalColumn", "config": {"key": "movieId", "number_buckets": 1001, "default_value": null}}, "dimension": 10, "combiner": "mean", "initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.31622776601683794, "seed": null, "dtype": "float32"}}, "ckpt_to_load_from": null, "tensor_name_in_ckpt": null, "max_norm": null, "trainable": true, "use_safe_embedding_lookup": true}}], "partitioner": null}, "build_input_shape": {"movieId": {"class_name": "TensorShape", "items": [null]}, "userId": {"class_name": "TensorShape", "items": [null]}}, "_is_feature_layer": true}
�	
_feature_columns

_resources
&"userId_embedding/embedding_weights
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "DenseFeatures", "name": "dense_features_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_features_1", "trainable": true, "dtype": "float32", "feature_columns": [{"class_name": "EmbeddingColumn", "config": {"categorical_column": {"class_name": "IdentityCategoricalColumn", "config": {"key": "userId", "number_buckets": 30001, "default_value": null}}, "dimension": 10, "combiner": "mean", "initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.31622776601683794, "seed": null, "dtype": "float32"}}, "ckpt_to_load_from": null, "tensor_name_in_ckpt": null, "max_norm": null, "trainable": true, "use_safe_embedding_lookup": true}}], "partitioner": null}, "build_input_shape": {"movieId": {"class_name": "TensorShape", "items": [null]}, "userId": {"class_name": "TensorShape", "items": [null]}}, "_is_feature_layer": true}
�
	variables
trainable_variables
regularization_losses
 	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 10]}, {"class_name": "TensorShape", "items": [null, 10]}]}
�

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
�

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
�

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
�
3iter

4beta_1

5beta_2
	6decay
7learning_ratemtmu!mv"mw'mx(my-mz.m{v|v}!v~"v'v�(v�-v�.v�"
	optimizer
X
0
1
!2
"3
'4
(5
-6
.7"
trackable_list_wrapper
X
0
1
!2
"3
'4
(5
-6
.7"
trackable_list_wrapper
 "
trackable_list_wrapper
�

8layers

	variables
9layer_metrics
trainable_variables
regularization_losses
:layer_regularization_losses
;non_trainable_variables
<metrics
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
"
_generic_user_object
3:1	�
2 dense_features/embedding_weights
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�

=layers
	variables
>layer_metrics
trainable_variables
regularization_losses
?layer_regularization_losses
@non_trainable_variables
Ametrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
"
_generic_user_object
6:4
��
2"dense_features_1/embedding_weights
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�

Blayers
	variables
Clayer_metrics
trainable_variables
regularization_losses
Dlayer_regularization_losses
Enon_trainable_variables
Fmetrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

Glayers
	variables
Hlayer_metrics
trainable_variables
regularization_losses
Ilayer_regularization_losses
Jnon_trainable_variables
Kmetrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:
2dense/kernel
:
2
dense/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
�

Llayers
#	variables
Mlayer_metrics
$trainable_variables
%regularization_losses
Nlayer_regularization_losses
Onon_trainable_variables
Pmetrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :

2dense_1/kernel
:
2dense_1/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
�

Qlayers
)	variables
Rlayer_metrics
*trainable_variables
+regularization_losses
Slayer_regularization_losses
Tnon_trainable_variables
Umetrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :
2dense_2/kernel
:2dense_2/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
�

Vlayers
/	variables
Wlayer_metrics
0trainable_variables
1regularization_losses
Xlayer_regularization_losses
Ynon_trainable_variables
Zmetrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
[0
\1
]2
^3"
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
�
	_total
	`count
a	variables
b	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�
	ctotal
	dcount
e
_fn_kwargs
f	variables
g	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
�"
htrue_positives
itrue_negatives
jfalse_positives
kfalse_negatives
l	variables
m	keras_api"�!
_tf_keras_metric�!{"class_name": "AUC", "name": "auc", "dtype": "float32", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}
�"
ntrue_positives
otrue_negatives
pfalse_positives
qfalse_negatives
r	variables
s	keras_api"�!
_tf_keras_metric�!{"class_name": "AUC", "name": "auc_1", "dtype": "float32", "config": {"name": "auc_1", "dtype": "float32", "num_thresholds": 200, "curve": "PR", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}
:  (2total
:  (2count
.
_0
`1"
trackable_list_wrapper
-
a	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
c0
d1"
trackable_list_wrapper
-
f	variables"
_generic_user_object
:� (2true_positives
:� (2true_negatives
 :� (2false_positives
 :� (2false_negatives
<
h0
i1
j2
k3"
trackable_list_wrapper
-
l	variables"
_generic_user_object
:� (2true_positives
:� (2true_negatives
 :� (2false_positives
 :� (2false_negatives
<
n0
o1
p2
q3"
trackable_list_wrapper
-
r	variables"
_generic_user_object
8:6	�
2'Adam/dense_features/embedding_weights/m
;:9
��
2)Adam/dense_features_1/embedding_weights/m
#:!
2Adam/dense/kernel/m
:
2Adam/dense/bias/m
%:#

2Adam/dense_1/kernel/m
:
2Adam/dense_1/bias/m
%:#
2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
8:6	�
2'Adam/dense_features/embedding_weights/v
;:9
��
2)Adam/dense_features_1/embedding_weights/v
#:!
2Adam/dense/kernel/v
:
2Adam/dense/bias/v
%:#

2Adam/dense_1/kernel/v
:
2Adam/dense_1/bias/v
%:#
2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
�2�
-__inference_functional_1_layer_call_fn_212532
-__inference_functional_1_layer_call_fn_212581
-__inference_functional_1_layer_call_fn_213011
-__inference_functional_1_layer_call_fn_213033�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_functional_1_layer_call_and_return_conditional_losses_212482
H__inference_functional_1_layer_call_and_return_conditional_losses_212801
H__inference_functional_1_layer_call_and_return_conditional_losses_212989
H__inference_functional_1_layer_call_and_return_conditional_losses_212455�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
!__inference__wrapped_model_211970�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *Z�W
U�R
(
movieId�
movieId���������
&
userId�
userId���������
�2�
/__inference_dense_features_layer_call_fn_213211
/__inference_dense_features_layer_call_fn_213219�
���
FullArgSpecE
args=�:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
J__inference_dense_features_layer_call_and_return_conditional_losses_213118
J__inference_dense_features_layer_call_and_return_conditional_losses_213203�
���
FullArgSpecE
args=�:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
1__inference_dense_features_1_layer_call_fn_213405
1__inference_dense_features_1_layer_call_fn_213397�
���
FullArgSpecE
args=�:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
L__inference_dense_features_1_layer_call_and_return_conditional_losses_213304
L__inference_dense_features_1_layer_call_and_return_conditional_losses_213389�
���
FullArgSpecE
args=�:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
,__inference_concatenate_layer_call_fn_213418�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_concatenate_layer_call_and_return_conditional_losses_213412�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_dense_layer_call_fn_213438�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_dense_layer_call_and_return_conditional_losses_213429�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_1_layer_call_fn_213458�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_1_layer_call_and_return_conditional_losses_213449�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_2_layer_call_fn_213478�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_2_layer_call_and_return_conditional_losses_213469�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
9B7
$__inference_signature_wrapper_212613movieIduserId�
!__inference__wrapped_model_211970�!"'(-.d�a
Z�W
U�R
(
movieId�
movieId���������
&
userId�
userId���������
� "1�.
,
dense_2!�
dense_2����������
G__inference_concatenate_layer_call_and_return_conditional_losses_213412�Z�W
P�M
K�H
"�
inputs/0���������

"�
inputs/1���������

� "%�"
�
0���������
� �
,__inference_concatenate_layer_call_fn_213418vZ�W
P�M
K�H
"�
inputs/0���������

"�
inputs/1���������

� "�����������
C__inference_dense_1_layer_call_and_return_conditional_losses_213449\'(/�,
%�"
 �
inputs���������

� "%�"
�
0���������

� {
(__inference_dense_1_layer_call_fn_213458O'(/�,
%�"
 �
inputs���������

� "����������
�
C__inference_dense_2_layer_call_and_return_conditional_losses_213469\-./�,
%�"
 �
inputs���������

� "%�"
�
0���������
� {
(__inference_dense_2_layer_call_fn_213478O-./�,
%�"
 �
inputs���������

� "�����������
L__inference_dense_features_1_layer_call_and_return_conditional_losses_213304�~�{
t�q
g�d
1
movieId&�#
features/movieId���������
/
userId%�"
features/userId���������

 
p
� "%�"
�
0���������

� �
L__inference_dense_features_1_layer_call_and_return_conditional_losses_213389�~�{
t�q
g�d
1
movieId&�#
features/movieId���������
/
userId%�"
features/userId���������

 
p 
� "%�"
�
0���������

� �
1__inference_dense_features_1_layer_call_fn_213397�~�{
t�q
g�d
1
movieId&�#
features/movieId���������
/
userId%�"
features/userId���������

 
p
� "����������
�
1__inference_dense_features_1_layer_call_fn_213405�~�{
t�q
g�d
1
movieId&�#
features/movieId���������
/
userId%�"
features/userId���������

 
p 
� "����������
�
J__inference_dense_features_layer_call_and_return_conditional_losses_213118�~�{
t�q
g�d
1
movieId&�#
features/movieId���������
/
userId%�"
features/userId���������

 
p
� "%�"
�
0���������

� �
J__inference_dense_features_layer_call_and_return_conditional_losses_213203�~�{
t�q
g�d
1
movieId&�#
features/movieId���������
/
userId%�"
features/userId���������

 
p 
� "%�"
�
0���������

� �
/__inference_dense_features_layer_call_fn_213211�~�{
t�q
g�d
1
movieId&�#
features/movieId���������
/
userId%�"
features/userId���������

 
p
� "����������
�
/__inference_dense_features_layer_call_fn_213219�~�{
t�q
g�d
1
movieId&�#
features/movieId���������
/
userId%�"
features/userId���������

 
p 
� "����������
�
A__inference_dense_layer_call_and_return_conditional_losses_213429\!"/�,
%�"
 �
inputs���������
� "%�"
�
0���������

� y
&__inference_dense_layer_call_fn_213438O!"/�,
%�"
 �
inputs���������
� "����������
�
H__inference_functional_1_layer_call_and_return_conditional_losses_212455�!"'(-.l�i
b�_
U�R
(
movieId�
movieId���������
&
userId�
userId���������
p

 
� "%�"
�
0���������
� �
H__inference_functional_1_layer_call_and_return_conditional_losses_212482�!"'(-.l�i
b�_
U�R
(
movieId�
movieId���������
&
userId�
userId���������
p 

 
� "%�"
�
0���������
� �
H__inference_functional_1_layer_call_and_return_conditional_losses_212801�!"'(-.z�w
p�m
c�`
/
movieId$�!
inputs/movieId���������
-
userId#� 
inputs/userId���������
p

 
� "%�"
�
0���������
� �
H__inference_functional_1_layer_call_and_return_conditional_losses_212989�!"'(-.z�w
p�m
c�`
/
movieId$�!
inputs/movieId���������
-
userId#� 
inputs/userId���������
p 

 
� "%�"
�
0���������
� �
-__inference_functional_1_layer_call_fn_212532�!"'(-.l�i
b�_
U�R
(
movieId�
movieId���������
&
userId�
userId���������
p

 
� "�����������
-__inference_functional_1_layer_call_fn_212581�!"'(-.l�i
b�_
U�R
(
movieId�
movieId���������
&
userId�
userId���������
p 

 
� "�����������
-__inference_functional_1_layer_call_fn_213011�!"'(-.z�w
p�m
c�`
/
movieId$�!
inputs/movieId���������
-
userId#� 
inputs/userId���������
p

 
� "�����������
-__inference_functional_1_layer_call_fn_213033�!"'(-.z�w
p�m
c�`
/
movieId$�!
inputs/movieId���������
-
userId#� 
inputs/userId���������
p 

 
� "�����������
$__inference_signature_wrapper_212613�!"'(-._�\
� 
U�R
(
movieId�
movieId���������
&
userId�
userId���������"1�.
,
dense_2!�
dense_2���������