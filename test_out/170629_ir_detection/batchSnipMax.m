function batchSnipMax(folder_in, folder_out, ext, left_x, right_x, down_y, up_y)
if nargin == 3
    left_x = 100;
    right_x = 100;
    down_y = 100;
    up_y = 100;
end
files = dir(strcat(folder_in, '\*.', ext));
for file = files'
    snip = snipMax(imread(file.name), left_x, right_x, down_y, up_y);
    imwrite(snip, strcat(folder_out, '\', erase(file.name, strcat('.', ext)), '_snip.', ext));
    save(strcat(folder_out, '\', erase(file.name,strcat('.', ext)), "_snip.mat"), "snip");
end