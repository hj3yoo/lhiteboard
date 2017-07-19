function img_out = snipMax(img_in, left_x, right_x, down_y, up_y)
if size(img_in,3) == 3
    img_in = rgb2gray(img_in);
end
brightest = find(img_in == max(max(img_in)));
img_out = [];
point = brightest(1)';
x = mod(point, size(img_in,1));
y = floor(point/ size(img_in,1) + 1);
img_out = [img_out img_in(max(1, x - left_x) : min(size(img_in, 1), x) + right_x, ...
    max(1, y - down_y) : min(size(img_in, 1), y + up_y))];
%for point = brightest'
%    x = mod(point, size(img_in,1));
%    y = floor(point/ size(img_in,1) + 1);
%    img_out = [img_out img_in(max(1, x - left_x) : min(size(img_in, 1), x) + right_x, ...
%        max(1, y - down_y) : min(size(img_in, 1), y + up_y))];
%end
fprintf("brightest pixel at value %d\n", max(max(img_in)));
end