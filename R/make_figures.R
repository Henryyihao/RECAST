#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) >= 1) args[1] else "source_data"
out_dir <- if (length(args) >= 2) args[2] else "figs"
quick <- "--quick" %in% args
extra_lib <- Sys.getenv("RECAST_R_LIB", "")
if (nzchar(extra_lib)) .libPaths(c(extra_lib, .libPaths()))
suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
})
stopifnot(requireNamespace("svglite", quietly=TRUE), capabilities("cairo"))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "previews"), showWarnings = FALSE)
dir.create(file.path(out_dir, "source_data"), showWarnings = FALSE)
font <- "DejaVu Sans"
pal <- c("RECAST" = "#167580", "CL pipeline" = "#839BB3",
         "Chain ladder" = "#839BB3", "Latest report" = "#737373",
         "Naive" = "#737373", "New parameters only" = "#BA9464",
         "Settled context" = "#A6A6A6", "True-delay CL" = "#BA9464",
         "Intensity reference" = "#8C7C97")
domains <- c("kit", "dv", "chng_flu", "hosp_cov", "eia930", "respinow",
             "nssp", "nhsn", "macro_m", "rtdsm_q", "alfred_w")
domain_names <- c(kit="DE-Hosp",dv="US-CLI",chng_flu="US-Flu",hosp_cov="US-Hosp",
                  eia930="US-Grid",respinow="DE-RESP",nssp="US-NSSP",nhsn="US-NHSN",
                  macro_m="US-Macro-M",rtdsm_q="US-Macro-Q",alfred_w="US-UI")
grid_names <- c(D="days",W="weeks",M="months",Q="quarters",h="hours",H="hours")
base_theme <- theme_classic(base_size = 8, base_family = font) +
  theme(axis.line = element_line(linewidth = 0.3, colour = "#454545"),
        axis.ticks = element_line(linewidth = 0.3, colour = "#454545"),
        axis.text = element_text(size = 7.5, colour = "#303030"),
        axis.title = element_text(size = 8),
        legend.position = "bottom", legend.title = element_blank(),
        legend.text = element_text(size = 7.5), legend.key.width = unit(4.5, "mm"),
        legend.margin = margin(0, 0, 0, 0),
        strip.background = element_blank(), strip.text = element_text(size=8,face="bold"),
        plot.title = element_text(size=9,face="bold",hjust=0),
        plot.subtitle = element_text(size=7.5,colour="#555555"),
        plot.tag = element_text(size=10,face="bold"),
        plot.margin = margin(5,5,4,5))
theme_set(base_theme)
read_data <- function(name) read.csv(file.path(data_dir,name), check.names=FALSE)
write_source <- function(df, name) write.csv(df, file.path(out_dir,"source_data",paste0(name,".csv")), row.names=FALSE)
save_figure <- function(p, name, width=180, height=100) {
  w <- width/25.4; h <- height/25.4
  grDevices::cairo_pdf(file.path(out_dir,paste0(name,".pdf")),width=w,height=h,family=font)
  print(p); invisible(dev.off())
  grDevices::png(file.path(out_dir,"previews",paste0(name,".png")),width=width,height=height,units="mm",res=180,type="cairo",bg="white")
  print(p); invisible(dev.off())
  if (!quick) {
    svglite::svglite(file.path(out_dir,paste0(name,".svg")),width=w,height=h,system_fonts=list(sans=font))
    print(p); invisible(dev.off())
    grDevices::tiff(file.path(out_dir,paste0(name,".tiff")),width=width,height=height,units="mm",res=600,type="cairo",compression="lzw",bg="white")
    print(p); invisible(dev.off())
  }
  message("Saved ",name)
}
tagged <- function(p) p + plot_annotation(tag_levels="a")
main <- read_data("main_metrics.csv")
horizons <- read_data("main_horizons.csv")
labels_domain <- function(x) unname(domain_names[x])
add_domain <- function(d) {
  d$domain_label <- factor(labels_domain(d$domain), levels=labels_domain(domains))
  d$revision <- ifelse(d$domain %in% domains[1:5],"Heavy revisions","Light revisions")
  d$revision <- factor(d$revision,levels=c("Heavy revisions","Light revisions"))
  d
}

method_main <- c(v12_final="RECAST",abl2_light="New parameters only",twostage="CL pipeline",
                 plain_oracle="Settled context",cl="Chain ladder")
forecast_ref <- main |> filter(task=="forecast",method=="plain_naive") |> select(domain,ref=CRPS_s)
now_ref <- main |> filter(task=="nowcast",method=="latest_report") |> select(domain,ref=MASE)
forecast <- main |> filter(task=="forecast",method %in% names(method_main)) |>
  inner_join(forecast_ref,by="domain") |> mutate(change=100*(CRPS_s/ref-1),label=unname(method_main[method])) |> add_domain()
nowcast <- main |> filter(task=="nowcast",method %in% names(method_main)) |>
  inner_join(now_ref,by="domain") |> mutate(change=100*(MASE/ref-1),label=unname(method_main[method])) |> add_domain()
dot_panel <- function(d,title,xlab,method_levels) {
  d <- d |> filter(label %in% method_levels) |> mutate(label=factor(label,levels=method_levels),
      y=match(domain,rev(domains)) + (as.integer(label)-(length(method_levels)+1)/2)*0.16)
  ggplot(d,aes(change,y,colour=label,shape=label)) +
    geom_hline(yintercept=6.5,colour="#E3E3E3",linewidth=.35) +
    geom_vline(xintercept=0,colour="#B5B5B5",linewidth=.4) +
    geom_point(size=1.9,stroke=.45) +
    scale_y_continuous(breaks=seq_along(domains),labels=labels_domain(rev(domains)),expand=expansion(add=.45)) +
    scale_x_continuous(labels=function(x) paste0(x,"%"),expand=expansion(mult=c(.08,.1))) +
    scale_colour_manual(values=pal) + scale_shape_manual(values=c(16,15,17,1)) +
    guides(colour=guide_legend(nrow=2),shape=guide_legend(nrow=2)) +
    labs(title=title,x=xlab,y=NULL) + theme(axis.line.y=element_blank(),axis.ticks.y=element_blank())
}
stopifnot(nrow(forecast)>0,nrow(nowcast)>0)
p_main <- dot_panel(forecast,"Forecast","CRPS change vs real-time backbone",c("RECAST","CL pipeline","New parameters only","Settled context")) |
  dot_panel(nowcast,"Nowcast","MASE change vs latest report",c("RECAST","Chain ladder","New parameters only"))
p_main <- tagged(p_main) & theme(legend.position="bottom")
save_figure(p_main,"fig_main",height=108)
write_source(bind_rows(forecast,nowcast),"fig_main")

bb <- bind_rows(main,read_data("bolt_s_metrics.csv"),read_data("bolt_b_metrics.csv"),
                read_data("toto_metrics.csv"),read_data("timemoe_metrics.csv"))
adapted_tags <- c(chronos2="v12_final",bolt_s="bolt_s_v3",bolt_b="bolt_b_v3",toto="toto_v3",timemoe="timemoe_v3")
backbone_labels <- c(chronos2="Chronos-2",bolt_s="Bolt-S",bolt_b="Bolt-B",toto="Toto-2",timemoe="Time-MoE")
bb_ref <- bb |> filter((task=="forecast" & method=="plain_naive") |
                       (task=="nowcast" & method=="latest_report")) |>
  select(backbone,domain,task,ref=MASE)
bb_adapt <- bb |> filter(method==unname(adapted_tags[backbone])) |>
  inner_join(bb_ref,by=c("backbone","domain","task")) |> mutate(change=100*(MASE/ref-1))
bb_adapt <- bb_adapt |> mutate(backbone_label=factor(backbone_labels[backbone],levels=unname(backbone_labels)),
                               domain_label=factor(domain_names[domain],levels=rev(labels_domain(domains))))
heat_panel <- function(d,title) {
  ggplot(d,aes(backbone_label,domain_label,fill=change)) + geom_tile(colour="white",linewidth=.6) +
    geom_text(aes(label=ifelse(abs(change)<.5,"0",sprintf("%+.0f",change))),size=2.6,colour="#272727") +
    scale_fill_gradient2(low="#7CB9BD",mid="#FAFAFA",high="#D9A18F",midpoint=0,
                         limits=c(-50,50),oob=scales::squish,breaks=c(-50,-25,0,25,50),
                         labels=function(x) paste0(x,"%"),name="MASE change") +
    guides(fill=guide_colourbar(barwidth=unit(55,"mm"),barheight=unit(3,"mm"),title.position="top")) +
    labs(title=title,x=NULL,y=NULL) +
    theme(axis.line=element_blank(),axis.ticks=element_blank(),axis.text.x=element_text(size=7.5),
          legend.position="bottom",legend.title=element_text(size=7.5))
}
p_bb <- heat_panel(filter(bb_adapt,task=="forecast"),"Forecast") | heat_panel(filter(bb_adapt,task=="nowcast"),"Nowcast")
p_bb <- tagged(p_bb) + plot_layout(guides="collect") & theme(legend.position="bottom")
stopifnot(nrow(bb_adapt)==110)
save_figure(p_bb,"fig_backbones",height=104)
write_source(bb_adapt,"fig_backbones")

horizon_ref <- horizons |> filter(task=="forecast",method=="plain_naive") |> select(domain,horizon,ref=CRPS_s)
plot_methods <- c(v12_final="RECAST",twostage="CL pipeline",plain_oracle="Settled context")
hdat <- horizons |> filter(task=="forecast",method %in% names(plot_methods)) |>
  inner_join(horizon_ref,by=c("domain","horizon")) |> mutate(ratio=CRPS_s/ref,label=unname(plot_methods[method]))
hdat <- add_domain(hdat)
p_h <- ggplot(hdat,aes(horizon,ratio,colour=label,linetype=label)) +
  geom_hline(yintercept=1,colour="#BBBBBB",linewidth=.35) + geom_line(linewidth=.65) +
  facet_wrap(~domain_label,ncol=3,scales="free_x") + scale_colour_manual(values=pal) +
  scale_linetype_manual(values=c("CL pipeline"="longdash","RECAST"="solid","Settled context"="dotted")) +
  labs(x="Forecast horizon (domain grid steps)",y="CRPS / real-time backbone")
save_figure(p_h,"fig_horizon",height=174)
write_source(hdat,"fig_horizon")

cdat <- horizons |> filter(task=="forecast",domain %in% c("kit","dv","chng_flu","eia930"),
                          method %in% c("v12_final","plain_naive","twostage")) |>
  mutate(label=case_when(method=="v12_final"~"RECAST",method=="plain_naive"~"Naive",TRUE~"CL pipeline")) |> add_domain()
p_cov <- ggplot(cdat,aes(horizon,COV80,colour=label,linetype=label)) + geom_hline(yintercept=.8,linetype="dashed",colour="#A5A5A5",linewidth=.4) +
  geom_line(linewidth=.65) + facet_wrap(~domain_label,ncol=2,scales="free_x") +
  scale_colour_manual(values=pal) +
  scale_linetype_manual(values=c("CL pipeline"="longdash","RECAST"="solid","Naive"="dotted")) +
  scale_y_continuous(limits=c(0,1),breaks=c(0,.4,.8,1)) +
  labs(x="Forecast horizon (domain grid steps)",y="Central 80% coverage")
save_figure(p_cov,"fig_coverage",height=100)
write_source(cdat,"fig_coverage")

age_methods <- c(v12_final="RECAST",cl="Chain ladder")
adat <- horizons |> filter(task=="nowcast",domain %in% c("kit","dv","chng_flu","hosp_cov","respinow","nhsn","macro_m","eia930"),method %in% names(age_methods)) |>
  mutate(label=unname(age_methods[method]),age=report_age,
         domain_grid=paste0(domain_names[domain]," (",grid_names[freq],")"))
adat$domain_grid <- factor(adat$domain_grid,levels=unique(adat$domain_grid[order(match(adat$domain,domains))]))
p_age <- ggplot(adat,aes(age,MASE,colour=label,linetype=label)) + geom_line(linewidth=.65) +
  facet_wrap(~domain_grid,ncol=4,scales="free") + scale_colour_manual(values=pal) +
  scale_linetype_manual(values=c("RECAST"="solid","Chain ladder"="longdash")) +
  labs(x="Report age (grid steps)",y="Scaled absolute error")
save_figure(p_age,"fig_nowcast_age",height=95)
write_source(adat,"fig_nowcast_age")

mechanism_labels <- c(recast="RECAST",cl="Chain ladder",oracle_cl="True-delay CL",oracle="Intensity reference")
mage <- read_data("mechanism_age.csv") |>
  filter(method %in% names(mechanism_labels)) |>
  mutate(label=factor(mechanism_labels[method],levels=unname(mechanism_labels)))
mswitch <- read_data("mechanism_switch.csv") |>
  filter(method %in% c("recast","cl")) |>
  mutate(label=factor(mechanism_labels[method],levels=unname(mechanism_labels)))
mprobe <- read_data("mechanism_probe.csv") |>
  mutate(label=factor(probe,levels=c("recast","init","raw_ratio"),
                      labels=c("Adapted","Initial","Raw ratio")))
mechanism_lines <- c("RECAST"="solid","Chain ladder"="solid",
                     "True-delay CL"="dashed","Intensity reference"="dotdash")
p_ma <- ggplot(mage,aes(report_age,MASE,colour=label,linetype=label)) +
  geom_line(linewidth=.65) + scale_colour_manual(values=pal) +
  scale_linetype_manual(values=mechanism_lines) +
  guides(colour=guide_legend(nrow=1),linetype=guide_legend(nrow=1)) +
  labs(title="Known-delay test",x="Report age (days)",y="Scaled absolute error")
p_ms <- ggplot(mswitch,aes(steps_since_switch,MASE,colour=label,linetype=label)) +
  geom_line(linewidth=.65) + geom_point(size=1.1) +
  scale_colour_manual(values=pal) + scale_linetype_manual(values=mechanism_lines) +
  guides(colour="none",linetype="none") +
  labs(title="Delay-distribution switch",x="Days since switch",y="Scaled absolute error")
p_mp <- ggplot(mprobe,aes(label,R2,fill=label)) +
  geom_col(width=.6) + geom_text(aes(label=sprintf("%.2f",R2)),vjust=-.5,size=2.6) +
  scale_fill_manual(values=c("Adapted"=unname(pal["RECAST"]),"Initial"="#AEB9C3","Raw ratio"="#CEB795")) +
  guides(fill="none") +
  scale_y_continuous(limits=c(0,1),breaks=c(0,.5,1),expand=expansion(mult=c(0,.04))) +
  labs(title="Delay decodability",x=NULL,y=expression("Held-out "*R^2)) +
  theme(legend.position="none",axis.text.x=element_text(size=7))
p_mech <- p_ma + p_ms + p_mp + plot_layout(widths=c(1,1,.85),guides="collect") +
  plot_annotation(tag_levels="a") & theme(legend.position="bottom",plot.title=element_text(size=8.5,face="bold"))
save_figure(p_mech,"fig_mechanism",height=83)
write_source(mage,"fig_mechanism_age")
write_source(mswitch,"fig_mechanism_switch")
write_source(mprobe,"fig_mechanism_probe")

deff <- read_data("data_efficiency.csv") |>
  mutate(label=case_when(method=="RECAST"~"RECAST",method=="chain_ladder"~"Chain ladder",TRUE~"CL pipeline"),
         domain_label=factor(domain_names[domain],levels=c("DE-Hosp","US-CLI")),
         task_label=factor(task,levels=c("nowcast","forecast"),labels=c("Nowcast","Forecast")))
p_eff <- ggplot(deff,aes(C,MASE,colour=label,shape=label)) +
  geom_line(linewidth=.65) + geom_point(size=1.9,stroke=.45) +
  facet_wrap(vars(domain_label,task_label),ncol=2,scales="free_y") +
  scale_colour_manual(values=pal) + scale_shape_manual(values=c(15,15,16)) +
  scale_x_continuous(breaks=c(96,128,192,256)) +
  labs(x="Context length (days)",y="MASE")
save_figure(p_eff,"fig_dataeff",height=109)
write_source(deff,"fig_dataeff")

triangle <- read_data("illustration_triangle.csv")
streams <- read_data("illustration_streams.csv")
completion <- read_data("illustration_completion.csv") |> filter(scope=="domain_pooled")
visible_triangle <- triangle |> filter(visible %in% c(TRUE,"True","TRUE"))
p_tri <- ggplot(visible_triangle,aes(u-origin,report_age,fill=normalized_value)) +
  geom_raster() +
  scale_fill_gradient(low="#EFF3F6",high="#326F7C",name="Relative value",
                      limits=c(0,max(visible_triangle$normalized_value,na.rm=TRUE))) +
  geom_line(data=distinct(triangle,u,origin) |> mutate(boundary=pmin(origin-u,80)),
            aes(u-origin,boundary),inherit.aes=FALSE,colour="#444444",linewidth=.45) +
  scale_x_continuous(breaks=c(-150,-100,-50,0)) +
  scale_y_continuous(breaks=c(0,20,40,60,80),expand=expansion(mult=c(0,.02))) +
  guides(fill=guide_colourbar(barwidth=unit(32,"mm"),barheight=unit(2.5,"mm"),title.position="top")) +
  labs(title="Visible version triangle",x="Days relative to origin",y="Report age (days)") +
  theme(axis.line=element_blank(),axis.ticks=element_blank(),legend.title=element_text(size=7.5))
stream_labels <- c(settled_visible="Settled history",as_of="Latest report",
                   age_0="Age 0",age_7="Age 7",age_21="Age 21")
stream_palette <- c("Settled history"="#353535","Latest report"="#9A9894",
                    "Age 0"="#A5BED1","Age 7"="#508E99","Age 21"="#C7A681")
sdat <- streams |> filter(stream %in% names(stream_labels),mask %in% c(TRUE,"True","TRUE")) |>
  mutate(label=factor(stream_labels[stream],levels=unname(stream_labels)))
p_str <- ggplot(sdat,aes(relative_position,value,colour=label)) +
  geom_line(linewidth=.55) + geom_vline(xintercept=0,linetype="dotted",colour="#858585",linewidth=.4) +
  scale_colour_manual(values=stream_palette) +
  scale_x_continuous(limits=c(-159,28),breaks=c(-150,-100,-50,0,28)) +
  scale_y_continuous(labels=scales::label_number(scale=.001,suffix="k")) +
  guides(colour=guide_legend(nrow=2)) +
  labs(title="Time-aligned age views",x="Days relative to origin",y="Hospitalisations (7-day sum)")
p_com <- ggplot(completion,aes(report_age,median)) +
  geom_ribbon(aes(ymin=q10,ymax=q90),fill="#A5BED1",alpha=.4) +
  geom_line(colour="#326F7C",linewidth=.65) +
  geom_hline(yintercept=1,linetype="dashed",colour="#969696",linewidth=.35) +
  labs(title="Completion across the domain",x="Report age (days)",y="Report / settled value") +
  scale_y_continuous(breaks=c(0,.5,1))
p_ill <- p_tri + (p_str / p_com) + plot_layout(widths=c(1,1.2)) + plot_annotation(tag_levels="a")
save_figure(p_ill,"fig_illustration",height=115)
write_source(visible_triangle,"fig_illustration_triangle")
write_source(sdat,"fig_illustration_streams")
write_source(completion,"fig_illustration_completion")

example_data <- read_data("example_traces.csv")
example_selection <- read_data("example_selection.csv")
trace_labels <- c(settled="Settled target",as_of="Latest report",RECAST_nowcast="RECAST",
                  RECAST_forecast="RECAST",naive_forecast="Real-time forecast",chain_ladder_nowcast="Chain ladder")
trace_pal <- c("Settled target"="#333333","Latest report"="#A4A29E",
               "RECAST"=unname(pal["RECAST"]),"Real-time forecast"="#787E88","Chain ladder"=unname(pal["Chain ladder"]))
example_plot <- function(case,legend=FALSE) {
  s <- example_selection |> filter(case_id==case)
  d <- example_data |> filter(case_id==case,trace %in% names(trace_labels)) |>
    mutate(label=factor(trace_labels[trace],levels=unique(unname(trace_labels))))
  ggplot(d,aes(relative_position,q50,colour=label,group=trace)) +
    geom_ribbon(data=d |> filter(trace %in% c("RECAST_nowcast","RECAST_forecast")),
                aes(ymin=q10,ymax=q90),fill=pal["RECAST"],colour=NA,alpha=.13,show.legend=FALSE) +
    geom_line(aes(linetype=label),linewidth=.5,na.rm=TRUE) +
    geom_vline(xintercept=0,linetype="dotted",colour="#858585",linewidth=.35) +
    scale_colour_manual(values=trace_pal) +
    scale_linetype_manual(values=c("Settled target"="solid","Latest report"="solid",
                                  "RECAST"="solid","Real-time forecast"="dashed","Chain ladder"="dotdash")) +
    guides(colour=if(legend) guide_legend(nrow=2) else "none",linetype=if(legend) guide_legend(nrow=2) else "none") +
    labs(title=s$title,x="Grid steps relative to origin",y=NULL) +
    theme(plot.title=element_text(size=8,face="bold"))
}
p_ex <- ((example_plot(1,TRUE) | example_plot(2)) /
         (example_plot(3) | example_plot(4))) +
  plot_layout(guides="collect") + plot_annotation(tag_levels="a") & theme(legend.position="bottom")
save_figure(p_ex,"fig_examples",height=117)
if(5 %in% example_selection$case_id) {
  save_figure(example_plot(5,TRUE),"fig_failure_example",width=180,height=70)
}
write_source(example_data,"fig_examples")
write_source(example_selection,"fig_examples_selection")

paired_losses <- read_data("paired_forecast_losses.csv") |>
  filter(domain %in% c("chng_flu","alfred_w")) |>
  mutate(origin_date=as.Date(origin_date))
timeline <- paired_losses |>
  group_by(domain,origin_date) |>
  summarise(n_pairs=n(),loss_difference=sum(recast_crps_sum-naive_crps_sum),
            n_cells=sum(recast_count),.groups="drop") |>
  mutate(mean_difference=loss_difference/n_cells) |>
  arrange(domain,origin_date) |> group_by(domain) |>
  mutate(cumulative_difference=cumsum(loss_difference)) |> ungroup()
failure_panel <- function(domain,metric,title,y_label) {
  d <- timeline |> filter(.data$domain==.env$domain)
  ggplot(d,aes(origin_date,.data[[metric]])) +
    geom_hline(yintercept=0,colour="#A0A0A0",linewidth=.35) +
    {if(metric=="mean_difference") geom_point(colour="#557D92",size=1.2,alpha=.8)
     else geom_line(colour="#326F7C",linewidth=.6)} +
    scale_x_date(date_breaks=if(domain=="chng_flu") "1 year" else "4 years",date_labels="%Y",expand=expansion(mult=c(.03,.04))) +
    scale_y_continuous(labels=scales::label_number()) +
    labs(title=title,x=NULL,y=y_label)
}
p_fail <- ((failure_panel("chng_flu","mean_difference","US-Flu","Mean CRPS difference") |
            failure_panel("alfred_w","mean_difference","US-UI","Mean CRPS difference")) /
           (failure_panel("chng_flu","cumulative_difference",NULL,"Cumulative scaled loss") |
            failure_panel("alfred_w","cumulative_difference",NULL,"Cumulative scaled loss"))) +
  plot_annotation(tag_levels="a")
save_figure(p_fail,"fig_failure",height=106)
write_source(timeline,"fig_failure")

schematic_theme <- theme_void(base_family=font) +
  theme(plot.margin=margin(4,4,4,4),plot.background=element_rect(fill="white",colour=NA))
box_layer <- function(x,y,w,h,label,fill="#EEF1F3",size=2.65) {
  list(annotate("rect",xmin=x-w/2,xmax=x+w/2,ymin=y-h/2,ymax=y+h/2,
                fill=fill,colour="#8B969C",linewidth=.35),
       annotate("text",x=x,y=y,label=label,size=size,family=font,lineheight=1.15,colour="#25343A"))
}
arrow_layer <- function(x,y,xend,yend,dashed=FALSE) {
  annotate("segment",x=x,y=y,xend=xend,yend=yend,colour="#65757C",linewidth=.4,
           linetype=if(dashed) "dashed" else "solid",
           arrow=grid::arrow(length=unit(1.5,"mm"),type="closed"))
}
label_layer <- function(x,y,label,size=2.6,bold=FALSE,hjust=.5) {
  annotate("text",x=x,y=y,label=label,size=size,family=font,
           fontface=if(bold) "bold" else "plain",hjust=hjust,colour="#303B40")
}
p_arch <- ggplot() + coord_cartesian(xlim=c(0,18),ylim=c(0,10),clip="off") + schematic_theme +
  label_layer(.1,9.7,"a   Inference from visible reports",3.1,TRUE,0) +
  box_layer(1.15,7.4,2.2,1.7,"Visible version\ntriangle",fill="#EEF1F3") +
  box_layer(4,7.4,2.6,1.7,"Aligned age views\nValues, masks, ages\nRevision profile",fill="#E2EFF0") +
  box_layer(7,7.4,2.6,2.7,"Age inputs\n\nTime / cross-age\nattention\n\nAdapted encoder",fill="#E4EAF0") +
  box_layer(10.3,8.3,2.6,1.25,"Direct forecast\nFrozen head",fill="#EEF1F3") +
  box_layer(10.3,5.8,2.6,1.3,"Nowcast quantiles\nEdge correction\ngate",fill="#E2EFF0") +
  box_layer(13.5,5.8,2.6,1.3,"Repaired histories\nFrozen backbone\nPool quantiles",fill="#EEF1F3") +
  box_layer(13.5,8.3,2.6,1.25,"Forecast gate\nDirect + anchor",fill="#E2EFF0") +
  box_layer(16.6,8.3,2.6,1.25,"Forecast quantiles\nSettled future",fill="#F0E9E1") +
  arrow_layer(2.25,7.4,2.7,7.4) + arrow_layer(5.3,7.4,5.7,7.4) +
  arrow_layer(8.3,7.9,9,8.3) + arrow_layer(8.3,6.7,9,5.8) +
  arrow_layer(11.6,8.3,12.2,8.3) + arrow_layer(11.6,5.8,12.2,5.8) +
  arrow_layer(13.5,6.45,13.5,7.675) + arrow_layer(14.8,8.3,15.3,8.3) +
  label_layer(14.6,7.0,"Anchor",2.4) +
  label_layer(13.5,4.75,"Inference paths: 0.1 / 0.5 / 0.9",2.5) +
  label_layer(.1,3.65,"b   Training on sampled reporting processes",3.1,TRUE,0) +
  box_layer(2.4,1.9,4.6,1.6,"Base series + reporting prior\nVisible triangle\nSettled edge targets",fill="#F0E9E1") +
  box_layer(8.6,1.9,4,1.6,"Shared adaptation\nEdge reconstruction\nGate supervision",fill="#E2EFF0") +
  box_layer(15.25,1.9,5,1.6,"Frozen teacher\nSettled-context forecasts\nDistillation",fill="#EEF1F3") +
  arrow_layer(4.7,1.9,6.6,1.9) + arrow_layer(12.75,1.9,10.6,1.9,TRUE) +
  label_layer(8.6,.55,"Training anchor: detached nowcast median",2.55)
save_figure(p_arch,"fig_arch",height=100)

mechanism_nodes <- data.frame(
  x=c(1.8,5.4,9,12.6,16.2),
  probability=c(.35,.25,.10,.10,.20),
  title=c("Delayed\nreporting","Converging\nestimates","Adjusted\nfeeds","Benchmark\nrevisions","No\nrevision"),
  detail=c("Cumulative counts\nDrift and switches\nCalendar effects",
           "Correlated age noise\nDecaying bias and scale\nSparse revision ages",
           "Spikes and dropouts\nStale provisional values\nLater cleaned reports",
           "Vintage-level shifts\nCorrection at a later\nbenchmark date",
           "All report ages\nretain the settled\nvalue"))
p_prior <- ggplot() + coord_cartesian(xlim=c(0,18),ylim=c(0,10),clip="off") + schematic_theme +
  box_layer(9,8.6,13,1.3,"Settled base series\nReal crops (85%); synthetic GP, renewal and seasonal ARMA signals (15%)",fill="#E4EAF0") +
  annotate("segment",x=9,xend=9,y=7.95,yend=7.45,colour="#65757C",linewidth=.4) +
  annotate("segment",x=1.8,xend=16.2,y=7.45,yend=7.45,colour="#65757C",linewidth=.4)
for (i in seq_len(nrow(mechanism_nodes))) {
  n <- mechanism_nodes[i,]
  p_prior <- p_prior + arrow_layer(n$x,7.45,n$x,6.85) +
    box_layer(n$x,5.4,3.25,2.9,"",fill=if(i==5) "#EEF1F3" else "#E2EFF0") +
    label_layer(n$x,6.2,n$title,2.55,TRUE) +
    label_layer(n$x,5.45,paste0(round(100*n$probability),"%"),2.7,TRUE) +
    label_layer(n$x,4.6,n$detail,2.5) +
    arrow_layer(n$x,3.95,n$x,3.2)
}
p_prior <- p_prior +
  box_layer(9,2.5,17.2,1.4,"Compositional modifiers\nPublication lags, late releases, early-count removals, smoothing, ratios, benchmark changes and context truncation",fill="#F0E9E1",size=2.55) +
  arrow_layer(9,1.8,9,1.2) +
  box_layer(9,.65,17.2,1.1,"Random origin in the generated version triangle\nVisible age views; settled edge targets; frozen-backbone settled-context forecast targets",fill="#EEF1F3",size=2.55)
save_figure(p_prior,"fig_prior",height=91)
write_source(mechanism_nodes,"fig_prior_mechanisms")

capture.output(sessionInfo(),file=file.path(out_dir,"R_session_info.txt"))
