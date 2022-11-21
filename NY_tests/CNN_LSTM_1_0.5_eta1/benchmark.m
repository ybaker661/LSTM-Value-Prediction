
lstmny = load('arb_NY.mat');
lstmny = lstmny.arbNY;
lstmli = load('arb_LONGIL.mat');
lstmli = lstmli.arbLONGIL;
lstmn = load('arb_NORTH.mat');
lstmn = lstmn.arbNORTH;
lstmw = load('arb_WEST.mat');
lstmw = lstmw.arbWEST;
trueny = load('prof_NY.mat');
trueny = trueny.profNY;
trueli = load('prof_LONGIL.mat');
trueli = trueli.profNY;
truen = load('prof_NORTH.mat');
truen = truen.profNORTH;
truew = load('prof_WEST.mat');
truew = truew.profNY;
rlny = readtable('NYC.csv');
rlny = table2array(rlny(1:end, 5));
rlli = readtable('LONGIL.csv');
rlli = table2array(rlli(1:end, 5));
rln = readtable('NORTH.csv');
rln = table2array(rln(1:end, 5));
rlw = readtable('WEST.csv');
rlw = table2array(rlw(1:end, 5));
x = 1:1:104533;

fig = figure();
subplot(2,2,1)
l1=plot(x, trueny/1000, '-','Color',[1 0 0]);
hold on
l2=plot(x, lstmny(1:104533)/1000, 'Color',[0 0 1]);
l3 = plot(x, rlny(1:104533)/(24*1000), '--', 'Color', [0 1 0]);
title('NYC')
legend({'Perfect Prediction', 'Our LSTM', "RL"}, 'FontSize', 8)
legend('Location', 'northwest')
grid on


subplot(2,2,2)
l1=plot(x, trueli/1000, '-','Color',[1 0 0]);
hold on
l2=plot(x, lstmli(1:104533)/1000, 'Color',[0 0 1]);
l3 = plot(x, rlli(1:104533)/(24*1000), '--', 'Color', [0 1 0]);
title('LONGIL')
grid on

subplot(2,2,3)
l1=plot(x, truen/1000, '-','Color',[1 0 0]);
hold on
l2=plot(x, lstmn(1:104533)/1000, 'Color',[0 0 1]);
l3 = plot(x, rln(1:104533)/(24*1000), '--', 'Color', [0 1 0]);
title('NORTH')
grid on

subplot(2,2,4)
l1=plot(x, truew/1000, '-','Color',[1 0 0]);
hold on
l2=plot(x, lstmw(1:104533)/1000, 'Color',[0 0 1]);
l3 = plot(x, rlw(1:104533)/(24*1000), '--', 'Color', [0 1 0]);
title('WEST')
grid on



han=axes(fig,'visible','off'); 
han.XLabel.Visible='on';
han.YLabel.Visible='on';
ylabel(han,'Cumulative Profit [k$]','fontsize',13);
xlabel(han,'Time Step (5 mins)','fontsize',13);
% add a bit space to the figure
% add legend
