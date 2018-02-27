using TensorFlow
using Distributions

num_of_input_nodes = 1
num_of_hidden_nodes = 80
num_of_output_nodes = 1
length_of_sequences = 10
num_of_training_epochs = 5000
size_of_mini_batch = 100
num_of_prediction_epochs = 100
learning_rate = 0.01
forget_bias = 0.8
num_of_sample = 1000

sess = Session(Graph())

# function get_batch(batch_size, X, t)
#   rnum = [random.randint(0, len(X) - 1) for x in range(0, batch_size)]
#   rnum = rand(1:ndims(X), batch_size)
# 
#   xs = [[[[y] for y in X[r]]] for r in rnum]
#   ts = [[t[r]] for r in rnum]
#   return xs, ts
# end


function create_data(nb_of_samples, sequence_len)
  X = zeros(nb_of_samples, sequence_len)
  for row_idx in range(1, nb_of_samples)
    X[row_idx, :] = convert(Array{Int8, 1}, round.(rand(sequence_len)))
    # Create the targets for each sequence
  end

  t = sum(X, 2)
  return X, t
end

# function make_prediction(nb_of_samples)
#   sequence_len = 10
#   xs, ts = create_data(nb_of_samples, sequence_len)
#   return [[[y] for y in x] for x in xs], [[x] for x in ts]
# end


function inference(input_ph, istate_ph)
  # TODO: わからないので先延ばし
  # weight1_var = Variable(random_normal([num_of_input_nodes, num_of_hidden_nodes], stddev=0.1), name="weight1")
  # weight2_var = Variable(random_normal([num_of_hidden_nodes, num_of_output_nodes], stddev=0.1), name="weight2")
  # bias1_var = Variable(random_normal([num_of_hidden_nodes], stddev=0.1), name="bias1")
  # bias2_var = Variable(random_normal([num_of_output_nodes], stddev=0.1), name="bias2")
  weight1_var = Variable(zeros(num_of_input_nodes, num_of_hidden_nodes), name="weight1")
  weight2_var = Variable(zeros(num_of_hidden_nodes, num_of_output_nodes), name="weight2")
  bias1_var = Variable(zeros(num_of_hidden_nodes), name="bias1")
  bias2_var = Variable(zeros(num_of_output_nodes), name="bias2")

  in1 = transpose(input_ph, [1, 0, 2])
  in2 = reshape(in1, [-1, num_of_input_nodes])
  in3 = in2 * weight1_var + bias1_var
  in4 = split(1, length_of_sequences, in3)

  cell = nn.rnn_cell.LSTMCell(num_of_hidden_nodes, forget_bias=forget_bias)
  rnn_output, states_op = nn.rnn(cell, in4, initial_state=istate_ph)
  # output_op = rnn_output[-1] * weight2_var + bias2_var

  # Add summary ops to collect data
  w1_hist = histogram_summary("weights1", weight1_var)
  w2_hist = histogram_summary("weights2", weight2_var)
  b1_hist = histogram_summary("biases1", bias1_var)
  b2_hist = histogram_summary("biases2", bias2_var)
  output_hist = histogram_summary("output",  output_op)
  results = [weight1_var, weight2_var, bias1_var,  bias2_var]

  return output_op, states_op, results
end


function loss(output_op, supervisor_ph)
  square_error = reduce_mean(square(output_op - supervisor_ph))
  loss_op = square_error
  scalar_summary("loss", loss_op)

  return loss_op
end


# function training(loss_op)
#   training_op = optimizer.minimize(loss_op)
# 
#   return training_op
# end
# 
# function calc_accuracy(output_op, prints=False)
#     inputs, ts = make_prediction(num_of_prediction_epochs)
#     pred_dict = {
#         input_ph: inputs,
#         supervisor_ph: ts,
#         istate_ph: zeros(num_of_prediction_epochs, num_of_hidden_nodes * 2),
#     }
#     # 不安
#     output = run(sess, [output_op], feed_dict=pred_dict)
# 
#     if prints
#       for p, q in zip(output[1], ts)
#         print("output: %f, correct: %d" % (p, q))
#       end
#     end
# 
#     opt = abs(output - ts)[1]
#     total = sum([x[1] < 0.05 ? 1 : 0.05 for x in opt])
#     print("accuracy %f" % (total / float(length(ts))))
# 
#     print("calc_accuracy")
#     return output
# end

# function train()
#   for epoch in range(0, num_of_training_epochs)
#     inputs, supervisors = get_batch(size_of_mini_batch, X, t)
#     train_dict = {
#                   input_ph:      inputs,
#                   supervisor_ph: supervisors,
#                   istate_ph:     zeros(size_of_mini_batch, num_of_hidden_nodes * 2),
#                  }
#     run(sess, training_op, feed_dict=train_dict)
# 
#     if (epoch) % 100 == 0
#       summary_str, train_loss = run(sess, [summary_op, loss_op], feed_dict=train_dict)
#       print("train#%d, train loss: %e" % (epoch, train_loss))
#       summary_writer.add_summary(summary_str, epoch)
#       if (epoch) % 500 == 0
#         calc_accuracy(output_op)
# 
#         calc_accuracy(output_op, prints=True)
#         datas = sess.run(datas_op)
#         saver.save(sess, "model.ckpt")
#       end
#     end
#   end
# end

srand(0)
# set_random_seed(0)

optimizer = train.GradientDescentOptimizer(learning_rate)

X, t = create_data(num_of_sample, length_of_sequences)

input_ph = placeholder(Float32, name="input", shape=[nothing, length_of_sequences, num_of_input_nodes])
supervisor_ph = placeholder(Float32, name="supervisor", shape=[nothing, num_of_output_nodes])
istate_ph = placeholder(Float32, name="istate", shape=[nothing, num_of_hidden_nodes * 2])

output_op, states_op, datas_op = inference(input_ph, istate_ph)
loss_op = loss(output_op, supervisor_ph)
# training_op = training(loss_op)
# 
# summary_op = summary.merge_all()
# 
# saver = train.Saver()
# summary_writer = tf.summary.FileWriter("/tmp/tensorflow_log", graph=sess.graph)
# run(sess, global_variables_initializer())
# train()
