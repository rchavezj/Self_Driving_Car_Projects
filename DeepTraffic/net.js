
//<![CDATA[

// a few things don't have var in front of them - they update already existing variables the game needs
lanesSide = 3;
patchesAhead = 50;
patchesBehind = 10;
trainIterations = 500000;

var num_inputs = (lanesSide * 2 + 1) * (patchesAhead + patchesBehind);
var num_actions = 5;
var temporal_window = 0;
var network_size = num_inputs * temporal_window + num_actions * temporal_window + num_inputs;

var layer_defs = [];
layer_defs.push({
    type: 'input',
    out_sx: 1,
    out_sy: 1,
    out_depth: network_size
});
layer_defs.push({
    type: 'fc',
    num_neurons: 36,
    activation: 'tanh'
});
layer_defs.push({
    type: 'fc',
    num_neurons: 24,
    activation: 'tanh'
});
layer_defs.push({
    type: 'fc',
    num_neurons: 24,
    activation: 'tanh'
});
layer_defs.push({
    type: 'fc',
    num_neurons: 24,
    activation: 'tanh'
});
layer_defs.push({
    type: 'regression',
    num_neurons: num_actions
});

var tdtrainer_options = {
    learning_rate: 0.001,
    momentum: 0.0,
    batch_size: 128,
    l2_decay: 0.01
};

var opt = {};
opt.temporal_window = temporal_window;
opt.experience_size = 100000;
opt.start_learn_threshold = 50000;
opt.gamma = 0.98;
opt.learning_steps_total = 500000;
opt.learning_steps_burnin = 1000;
opt.epsilon_min = 0.0;
opt.epsilon_test_time = 0.0;
opt.layer_defs = layer_defs;
opt.tdtrainer_options = tdtrainer_options;

brain = new deepqlearn.Brain(num_inputs, num_actions, opt);

learn = function (state, lastReward) {
    brain.backward(lastReward);
    var action = brain.forward(state);

    draw_net();
    draw_stats();

    return action;
}
/*###########*/
if (brain) {
brain.value_net.fromJSON({"layers":[{"out_depth":19,"out_sx":1,"out_sy":1,"layer_type":"input"},{"out_depth":1,"out_sx":1,"out_sy":1,"layer_type":"fc","num_inputs":19,"l1_decay_mul":0,"l2_decay_mul":1,"filters":[{"sx":1,"sy":1,"depth":19,"w":{"0":-0.3910240786703292,"1":0.10121960741805189,"2":-0.34015831223457393,"3":-0.06888494885521214,"4":-0.07102692071002947,"5":-0.4837813807769807,"6":0.118645569410263,"7":-0.07837247059199196,"8":-0.15495166846447606,"9":-0.24321636221986778,"10":-0.035135182110609564,"11":-0.003944880196269991,"12":-0.023329875951041388,"13":0.07672805208434852,"14":0.047168299933024106,"15":-0.2760531850268392,"16":0.2196387939495988,"17":-0.004582699447506437,"18":0.22730040540047752}}],"biases":{"sx":1,"sy":1,"depth":1,"w":{"0":0.1}}},{"out_depth":1,"out_sx":1,"out_sy":1,"layer_type":"relu"},{"out_depth":5,"out_sx":1,"out_sy":1,"layer_type":"fc","num_inputs":1,"l1_decay_mul":0,"l2_decay_mul":1,"filters":[{"sx":1,"sy":1,"depth":1,"w":{"0":-0.8070935819900138}},{"sx":1,"sy":1,"depth":1,"w":{"0":0.056782177104611455}},{"sx":1,"sy":1,"depth":1,"w":{"0":1.018613249879754}},{"sx":1,"sy":1,"depth":1,"w":{"0":1.3639320672991124}},{"sx":1,"sy":1,"depth":1,"w":{"0":-1.396776426008625}}],"biases":{"sx":1,"sy":1,"depth":5,"w":{"0":0,"1":0,"2":0,"3":0,"4":0}}},{"out_depth":5,"out_sx":1,"out_sy":1,"layer_type":"regression","num_inputs":5}]});
}
/*###########*/
if (brain) {
brain.value_net.fromJSON({"layers":[{"out_depth":19,"out_sx":1,"out_sy":1,"layer_type":"input"},{"out_depth":1,"out_sx":1,"out_sy":1,"layer_type":"fc","num_inputs":19,"l1_decay_mul":0,"l2_decay_mul":1,"filters":[{"sx":1,"sy":1,"depth":19,"w":{"0":-0.39038720239175634,"1":0.09941770700051501,"2":-0.29782529500029037,"3":-0.11087713882730339,"4":-0.11504197129426195,"5":-0.482288488931771,"6":0.08489286909395212,"7":-0.08005729478904013,"8":-0.11953272337085653,"9":-0.1394032061373678,"10":-0.0334803010103356,"11":-0.07157089913015918,"12":-0.08450081346123009,"13":0.02770105282934759,"14":0.037091210458726445,"15":-0.23615254516037465,"16":0.3116757161521776,"17":-0.03428415593035474,"18":0.1010448707297727}}],"biases":{"sx":1,"sy":1,"depth":1,"w":{"0":0.033244878021594086}}},{"out_depth":1,"out_sx":1,"out_sy":1,"layer_type":"relu"},{"out_depth":5,"out_sx":1,"out_sy":1,"layer_type":"fc","num_inputs":1,"l1_decay_mul":0,"l2_decay_mul":1,"filters":[{"sx":1,"sy":1,"depth":1,"w":{"0":-0.658930130883463}},{"sx":1,"sy":1,"depth":1,"w":{"0":0.18609050819404901}},{"sx":1,"sy":1,"depth":1,"w":{"0":1.0901283507449466}},{"sx":1,"sy":1,"depth":1,"w":{"0":1.4320411270239983}},{"sx":1,"sy":1,"depth":1,"w":{"0":-1.1797379920825315}}],"biases":{"sx":1,"sy":1,"depth":5,"w":{"0":48.12070160712304,"1":48.11552304316431,"2":48.10205251135444,"3":48.117734452802345,"4":48.11842637582941}}},{"out_depth":5,"out_sx":1,"out_sy":1,"layer_type":"regression","num_inputs":5}]});
}