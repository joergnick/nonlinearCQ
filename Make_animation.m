
clear all
load('Fractional_h05_dof1000')
X_mesh3(:,:)=plot_grid(1,:,:);
Y_mesh3(:,:)=plot_grid(2,:,:);
n_grid3=sqrt(length(u_ges(:,1))) ; 
u_ges3=u_ges;
max(max(u_ges3))

u_sq3=zeros(n_grid3,n_grid3);
n_grid3=sqrt(length(u_ges(:,1)));
[scatterer3 ] = create_scatterer(X_mesh3,Y_mesh3);

[u_ges3 ] = draw_scatterer( u_ges3, scatterer3);
load('Acoustic_h05_dof100')
X_mesh2(:,:)=plot_grid(1,:,:);
Y_mesh2(:,:)=plot_grid(2,:,:);
n_grid2=sqrt(length(u_ges(:,1))) ; 
u_ges2=u_ges;

u_sq2=zeros(n_grid2,n_grid2);
n_grid2=sqrt(length(u_ges(:,1)));
[scatterer2 ] = create_scatterer(X_mesh2,Y_mesh2);

[u_ges2 ] = draw_scatterer( u_ges2, scatterer2);

load('GIBCe4_h05_dof1000')


n_grid=sqrt(length(u_ges(:,1)));
u_sq=zeros(n_grid,n_grid);

X_mesh=zeros(n_grid,n_grid);

Y_mesh=zeros(n_grid,n_grid);


X_mesh(:,:)=plot_grid(1,:,:);
Y_mesh(:,:)=plot_grid(2,:,:);
  
% X_mesh=double_resolution(X_mesh);
% Y_mesh=double_resolution(Y_mesh);
%u_ges=double_resolution(u_ges);

[scatterer ] = create_scatterer(X_mesh,Y_mesh);

[u_ges ] = draw_scatterer( u_ges, scatterer);

%u_ges=normalize_u(u_ges);
%u_ges=draw_magnet(plot_grid,u_ges);

figure('Position',[200 200 1300 350])

for j=1:300
    limit_colors=[0,1.5];
    limit_height=[-2.5,2.5];
    limit_x=[-0.25,1.25];
    limit_y=[-0.75,0.75];
    width=0.213405797101449;
    colormap parula(50)
    u_long=u_ges(:,j);
    u_long2=u_ges2(:,j); 
    u_long3=u_ges3(:,j); 
    for i=1:n_grid
        
        u_sq(:,i)=u_long((i-1)*n_grid+1:i*n_grid);
        
    end
    for i=1:n_grid2
       u_sq2(:,i)=u_long2((i-1)*n_grid2+1:i*n_grid2);
    end
    for i=1:n_grid3
       u_sq3(:,i)=u_long3((i-1)*n_grid3+1:i*n_grid3);
    end
    startp=1;
    endp=n_grid;
    
    subplot(1,3,1)
   
    surf(X_mesh(startp:endp,:),Y_mesh(startp:endp,:),u_sq(startp:endp,:)','edgecolor','none')
   
    caxis(limit_colors)
   % view(180,0)
    %view(0,90)
     view(2)
    xlim(limit_x)
    ylim(limit_y)
    zlim(limit_height)
    
    startp2=1;
    endp2=n_grid2;

 %   hsp1 = get(gca, 'Position') ;      
   %% Position Subplot 1    
%   set(gca, 'Position', [hsp1(1)-0.05 hsp1(2) width+0.02 hsp1(4)]) 
%     subplot(1,3,2)
%      
%        
%     surf(X_mesh2(startp2:endp2,:),Y_mesh2(startp2:endp2,:),u_sq2(startp2:endp2,:)','edgecolor','none')
%  
%     colorbar
%     caxis(limit_colors)
%    view(180,0)
%     view(2)
%     xlim(limit_x)
%     ylim(limit_y)
%     zlim(limit_height)
%     
%     startp3=1;
%     endp3=n_grid3;
%     
%     hsp2 = get(gca, 'Position') ;
%        % Position Subplot 2 
%     set(gca, 'Position', [hsp2(1)-0.04 hsp1(2) width+0.03 hsp1(4)]) 
%  title('Acoustic b.c.')
%      subplot(1,3,3)
%      hsp3=get(gca,'Position');
%     surf(X_mesh3(startp3:endp3,:),Y_mesh3(startp3:endp3,:),u_sq3(startp3:endp3,:)','edgecolor','none')
%     colorbar
%     
%     
%     caxis(limit_colors)
%     view(0,180)
%     view(2)
%     xlim(limit_x)
%     ylim(limit_y)
%     zlim(limit_height)
%      title('Fractional absorbing b.c.')
%      colorbar
%      hsp2(3)
%       %% Position Subplot 3  
%      set(gca, 'Position', [hsp3(1)-0.03 hsp3(2) width+0.02 hsp1(4)]) 
    drawnow
end