	m<?b????m<?b????!m<?b????	@?e?O$;@@?e?O$;@!@?e?O$;@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:m<?b?????Q????A?(B?v???Y;S???.??rEagerKernelExecute 0*	!?rh?Ab@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???[???!?H??M?D@)*?t??1kz0wbB@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatS?'?ݢ?!CK?:9@)????5"??1?L^?\?5@:Preprocessing2U
Iterator::Model::ParallelMapV2-??淪?!?z4ML.@)-??淪?1?z4ML.@:Preprocessing2F
Iterator::Model?H?5C??!X7?+1l8@) ?t?????1??K?"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip6?!?A??!*r???R@)?e?%⭃?1?=??3Q@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?~4?27?!tf?X?@)?~4?27?1tf?X?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor? ݗ3?u?!??f?:@)? ݗ3?u?1??f?:@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??m?R]??!???fj?E@){O崧?l?1QT??Q@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 27.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2t10.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9A?e?O$;@I??f?6R@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Q?????Q????!?Q????      ??!       "      ??!       *      ??!       2	?(B?v????(B?v???!?(B?v???:      ??!       B      ??!       J	;S???.??;S???.??!;S???.??R      ??!       Z	;S???.??;S???.??!;S???.??b      ??!       JCPU_ONLYYA?e?O$;@b q??f?6R@