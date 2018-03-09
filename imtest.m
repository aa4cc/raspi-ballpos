close all;
clear all;

%%
obj = 'Detector-Red';
%obj = 'Detector-Green';
%obj = 'Detector-Blue';

dwnsample = RaspiImage('147.32.86.182', 1150, obj, 'whole');
figure(1)
imagesc(dwnsample)
colorbar


roi = RaspiImage('147.32.86.182', 1150, obj, 'roi');
figure(2)
imagesc(roi)
colorbar

%%
% RGB image
figure(3)
rgb = RaspiImage('147.32.86.182', 1150, 'Processor', 'any');
r = rgb(:,:,1);
g = rgb(:,:,2);
b = rgb(:,:,3);

hsv = rgb2hsv(rgb);
hsv(:,:,1) = hsv(:,:,1)*360;
h = hsv(:,:,1);
s = hsv(:,:,2);
v = hsv(:,:,3);

imshow(rgb);

%%
figure(1)
clf()
subplot(1,2,1);
imshow(rgb)

figure(1)
subplot(1,2,1);
tol = 0.9;
while 1
    figure(1);
    p = round(ginput(1));
    sel_rgb = squeeze(rgb(p(2), p(1), :))';
    sel_hsv = squeeze(hsv(p(2), p(1), :))';

    figure(2);
    rectangle('Position',[0,0,1,1],'FaceColor',sel_rgb)
    disp(sel_rgb)
    disp(sel_hsv)
end
%%
kulicky = ((r+g+b)>150);
tst = (kulicky & ((0 < h) & (h < 20))); % Oranová
figure(1);
subplot(1,2,2);
imshow(tst)


