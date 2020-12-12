function ak_plot_cells_and_ues(UEpositions, BSpositions)
% function ak_plot_cells_and_ues(UEpositions, BSpositions)
%plots BS and UEs. Does not draw the cell hexagons.

should_plot_UE_number = 1;
num_BS = length(BSpositions);
num_UEs = size(UEpositions,1);

%if exist('handlerPlot','var') %delete previous users
%    delete(handlerPlot)
%end
%handlerPlot = plot(locX, locY,'x');
%handlerPlot = plot(squeeze(allMSlocations(:,:,1))', ...
%    squeeze(allMSlocations(:,:,2))','x');
handlerPlot = plot(real(UEpositions), imag(UEpositions),'x');
%plot(locX(1,:), locY(1,:),'o','MarkerSize',12)
%title('BS and UE locations')
%pause
drawnow

for i=1:num_BS
    bs_color = handlerPlot(i).Color;    
    text(real(BSpositions(i))-15,imag(BSpositions(i))-40,num2str(i),'Color',bs_color,'FontSize',16)    
    for j=1:num_UEs
        x = real(UEpositions(j,i)); %allMSlocations(i,j,1);
        delta_y = imag(UEpositions(j,i)); %delta_y = allMSlocations(i,j,2);
        if should_plot_UE_number == 1
            text(x,delta_y,num2str(j),'Color',bs_color)
        end
    end
end

axis equal
title('')
axis tight
