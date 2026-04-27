function session_plotter_matlab(sessionPath, varargin)
%SESSION_PLOTTER_MATLAB Playback/export tool for Braccio session logs.
%
% Usage examples:
%   session_plotter_matlab("logs/sessions/session_20260316_021937");
%   session_plotter_matlab("logs/sessions/session_20260316_021937", "SaveMP4", true);
%
% Requires:
%   - Robotics System Toolbox (`importrobot`, `show`)
%   - The session log folder produced by the Python controller

opts = parseInputs(varargin{:});
[sessionDir, records, meta] = loadSessionData(sessionPath);

if isempty(records)
    error("session_plotter_matlab:EmptySession", "No records found in %s", sessionDir);
end

urdfPath = "";
if isfield(meta, "urdf_path")
    urdfPath = string(meta.urdf_path);
end
if strlength(urdfPath) == 0 || ~isfile(urdfPath)
    urdfPath = fullfile(fileparts(fileparts(mfilename("fullpath"))), "Tinkerkit_model", "tinkerkit.urdf");
end

robot = importrobot(char(urdfPath));
robot.DataFormat = "row";
robot.Gravity = [0 0 -9.81];

times = zeros(numel(records), 1);
t0 = getFieldOr(records{1}, "host_monotonic", 0.0);
for i = 1:numel(records)
    times(i) = getFieldOr(records{i}, "host_monotonic", t0) - t0;
end

dt = diff(times);
if isempty(dt)
    fps = 25;
else
    fps = max(5, min(60, round(1.0 / max(1e-3, median(dt)))));
end

toolPts = precomputeToolPoints(records);
feasibleCloud = [];
if isfield(meta, "feasible_cloud_path") && isfile(string(meta.feasible_cloud_path))
    feasibleCloud = readNpyMatrix(string(meta.feasible_cloud_path));
end

[severity, clearanceMm, speedScale, objectiveProxy] = precomputeMetrics(records, meta);

fig = figure( ...
    "Name", "Braccio Session Plotter (MATLAB)", ...
    "Color", [0.07 0.10 0.16], ...
    "Position", [80 60 1480 860] ...
);

tl = tiledlayout(fig, 2, 2, "TileSpacing", "compact", "Padding", "compact");
ax3d = nexttile(tl, [2 1]);
axMetric = nexttile(tl, 2);
axObjective = nexttile(tl, 4);

setupMainAxes(ax3d, toolPts, feasibleCloud);
if ~isempty(feasibleCloud)
    scatter3(ax3d, feasibleCloud(:, 1), feasibleCloud(:, 2), feasibleCloud(:, 3), 3, ...
        "MarkerFaceColor", [0.60 0.82 1.00], ...
        "MarkerEdgeColor", "none", ...
        "MarkerFaceAlpha", 0.035);
end

plot(axMetric, times, severity, "-", "Color", [0.95 0.62 0.15], "LineWidth", 1.4); hold(axMetric, "on");
plot(axMetric, times, clearanceMm, "-", "Color", [0.35 0.78 0.98], "LineWidth", 1.4);
plot(axMetric, times, speedScale .* 200.0, "-", "Color", [0.35 0.86 0.55], "LineWidth", 1.4);
metricCursor = xline(axMetric, times(1), "--", "Color", [0.98 0.98 0.98], "LineWidth", 1.0);
grid(axMetric, "on");
title(axMetric, "Planner Metrics");
xlabel(axMetric, "t [s]");
ylabel(axMetric, "mixed units");
legend(axMetric, {"severity", "clearance [mm]", "speed scale x200"}, "TextColor", [1 1 1], "Color", [0.11 0.14 0.20]);
set(axMetric, "Color", [0.10 0.13 0.19], "XColor", [1 1 1], "YColor", [1 1 1]);

plot(axObjective, times, objectiveProxy, "-", "Color", [0.72 0.82 1.00], "LineWidth", 1.5); hold(axObjective, "on");
objectiveCursor = xline(axObjective, times(1), "--", "Color", [0.98 0.98 0.98], "LineWidth", 1.0);
grid(axObjective, "on");
title(axObjective, "Objective Proxy");
xlabel(axObjective, "t [s]");
ylabel(axObjective, "score");
set(axObjective, "Color", [0.10 0.13 0.19], "XColor", [1 1 1], "YColor", [1 1 1]);

statusBox = annotation(fig, "textbox", [0.70 0.72 0.27 0.22], ...
    "String", "", ...
    "FitBoxToText", "off", ...
    "Color", [0.95 0.97 1.00], ...
    "FontName", "Consolas", ...
    "FontSize", 10, ...
    "EdgeColor", [0.15 0.20 0.28], ...
    "BackgroundColor", [0.07 0.10 0.16], ...
    "LineWidth", 1.0);

vid = [];
if opts.SaveMP4
    outPath = fullfile(sessionDir, "session_playback_matlab_720p.mp4");
    vid = VideoWriter(outPath, "MPEG-4");
    vid.FrameRate = fps;
    vid.Quality = 95;
    open(vid);
    fprintf("[INFO] Writing MP4 -> %s\n", outPath);
end

for idx = 1:numel(records)
    rec = records{idx};
    joints = getJointVector(rec);

    cla(ax3d);
    setupMainAxes(ax3d, toolPts, feasibleCloud);
    if ~isempty(feasibleCloud)
        scatter3(ax3d, feasibleCloud(:, 1), feasibleCloud(:, 2), feasibleCloud(:, 3), 3, ...
            "MarkerFaceColor", [0.60 0.82 1.00], ...
            "MarkerEdgeColor", "none", ...
            "MarkerFaceAlpha", 0.03);
        hold(ax3d, "on");
    else
        hold(ax3d, "on");
    end

    show(robot, joints, "Parent", ax3d, "PreservePlot", false, "Frames", "off", "Visuals", "on");
    hold(ax3d, "on");

    drawToFCoverage(ax3d, rec);
    drawObstacleGeometry(ax3d, rec);
    drawCommittedObstacles(ax3d, rec);
    drawBreadcrumb(ax3d, toolPts, times, idx);
    drawFuturePlan(ax3d, rec);
    plot3(ax3d, toolPts(idx, 1), toolPts(idx, 2), toolPts(idx, 3), "o", ...
        "MarkerFaceColor", [0.0 0.78 0.42], "MarkerEdgeColor", "none", "MarkerSize", 7);

    recTitle = sprintf('Braccio Session Playback | t = %.2f s | mode = %s', ...
        getFieldOr(rec, "wall_time", 0.0), char(string(getFieldOr(rec, "mode", ""))));
    title(ax3d, recTitle, "Color", [0.98 0.98 1.00]);

    statusBox.String = buildStatusText(rec);
    metricCursor.Value = times(idx);
    objectiveCursor.Value = times(idx);

    drawnow;
    if ~isempty(vid)
        frame = getframe(fig);
        writeVideo(vid, frame);
    else
        pause(max(0.001, 1.0 / fps));
    end
end

if ~isempty(vid)
    close(vid);
    fprintf("[INFO] MATLAB MP4 export complete.\n");
end

end

function opts = parseInputs(varargin)
    parser = inputParser;
    parser.addParameter("SaveMP4", false, @(x) islogical(x) || isnumeric(x));
    parse(parser, varargin{:});
    opts = parser.Results;
    opts.SaveMP4 = logical(opts.SaveMP4);
end

function [sessionDir, records, meta] = loadSessionData(sessionPath)
    sessionPath = string(sessionPath);
    if isfolder(sessionPath)
        sessionDir = char(sessionPath);
        dataPath = fullfile(sessionDir, "session.jsonl");
    else
        dataPath = char(sessionPath);
        sessionDir = fileparts(dataPath);
    end

    if ~isfile(dataPath)
        error("session_plotter_matlab:MissingFile", "Missing session file: %s", dataPath);
    end

    records = {};
    fid = fopen(dataPath, "r");
    cleanup = onCleanup(@() fclose(fid));
    while true
        line = fgetl(fid);
        if ~ischar(line)
            break;
        end
        line = strtrim(line);
        if strlength(line) == 0
            continue;
        end
        records{end + 1, 1} = jsondecode(line); %#ok<AGROW>
    end

    meta = struct();
    metaPath = fullfile(sessionDir, "meta.json");
    if isfile(metaPath)
        meta = jsondecode(fileread(metaPath));
    end
end

function value = getFieldOr(s, fieldName, defaultValue)
    if isstruct(s) && isfield(s, fieldName)
        value = s.(fieldName);
    else
        value = defaultValue;
    end
end

function joints = getJointVector(rec)
    joints = double(getFieldOr(rec, "joints_deg", [90 90 90 90 90 73]));
    joints = reshape(joints, 1, []);
end

function toolPts = precomputeToolPoints(records)
    n = numel(records);
    toolPts = zeros(n, 3);
    for i = 1:n
        rec = records{i};
        eef = getFieldOr(rec, "eef_m", [0 0 0]);
        toolPts(i, :) = reshape(double(eef(1:3)), 1, 3);
    end
end

function [severity, clearanceMm, speedScale, objectiveProxy] = precomputeMetrics(records, meta)
    n = numel(records);
    severity = zeros(n, 1);
    clearanceMm = nan(n, 1);
    speedScale = ones(n, 1);
    objectiveProxy = zeros(n, 1);
    thresholdMm = double(getFieldOr(meta, "threshold_mm", 200.0));

    for i = 1:n
        rec = records{i};
        planner = getFieldOr(rec, "planner_debug", struct());
        dbg = getFieldOr(planner, "debug", struct());
        severity(i) = double(getFieldOr(dbg, "severity", 0.0));
        clearanceMm(i) = double(getFieldOr(dbg, "min_clearance_mm", nan));
        speedScale(i) = double(getFieldOr(dbg, "speed_scale", 1.0));

        goal = getFieldOr(planner, "goal", struct());
        thetaErr = abs(double(getFieldOr(rec, "theta", 0.0)) - double(getFieldOr(goal, "theta", getFieldOr(rec, "theta", 0.0)))) / 180.0;
        rErr = abs(double(getFieldOr(rec, "r_mm", 0.0)) - double(getFieldOr(goal, "r_mm", getFieldOr(rec, "r_mm", 0.0)))) / 240.0;
        zErr = abs(double(getFieldOr(rec, "z_mm", 0.0)) - double(getFieldOr(goal, "z_mm", getFieldOr(rec, "z_mm", 0.0)))) / 250.0;

        if isnan(clearanceMm(i))
            clearancePenalty = 0.0;
        else
            clearancePenalty = max(0.0, thresholdMm - clearanceMm(i)) / max(1.0, thresholdMm);
        end

        objectiveProxy(i) = thetaErr + rErr + zErr + 0.8 * clearancePenalty + 0.2 * (1.0 - speedScale(i));
    end
end

function setupMainAxes(ax, toolPts, feasibleCloud)
    cla(ax);
    hold(ax, "on");
    axis(ax, "vis3d");
    view(ax, 42, 24);
    grid(ax, "on");
    ax.Color = [0.10 0.13 0.19];
    ax.XColor = [1 1 1];
    ax.YColor = [1 1 1];
    ax.ZColor = [1 1 1];
    xlabel(ax, "X [m]");
    ylabel(ax, "Y [m]");
    zlabel(ax, "Z [m]");

    if ~isempty(feasibleCloud)
        lo = min(feasibleCloud, [], 1) - 0.08;
        hi = max(feasibleCloud, [], 1) + 0.08;
    else
        lo = min(toolPts, [], 1) - 0.08;
        hi = max(toolPts, [], 1) + 0.08;
    end
    xlim(ax, [lo(1) hi(1)]);
    ylim(ax, [lo(2) hi(2)]);
    zlim(ax, [lo(3) hi(3)]);
end

function drawBreadcrumb(ax, toolPts, times, idx)
    tNow = times(idx);
    mask = times >= (tNow - 5.0) & times <= tNow;
    pts = toolPts(mask, :);
    if ~isempty(pts)
        plot3(ax, pts(:, 1), pts(:, 2), pts(:, 3), "-", "Color", [0.0 0.78 0.42], "LineWidth", 2.0);
    end
end

function drawFuturePlan(ax, rec)
    plan = getFieldOr(rec, "future_plan", []);
    if isempty(plan)
        return;
    end

    pts = [];
    way = [];
    for i = 1:numel(plan)
        entry = plan(i);
        eef = getFieldOr(entry, "eef_m", []);
        if numel(eef) < 3
            continue;
        end
        xyz = double(eef(1:3));
        pts(end + 1, :) = xyz; %#ok<AGROW>
        if logical(getFieldOr(entry, "waypoint", false))
            way(end + 1, :) = xyz; %#ok<AGROW>
        end
    end

    if ~isempty(pts)
        plot3(ax, pts(:, 1), pts(:, 2), pts(:, 3), "--", "Color", [0.49 0.83 0.99], "LineWidth", 1.8);
    end
    if ~isempty(way)
        scatter3(ax, way(:, 1), way(:, 2), way(:, 3), 28, ...
            "MarkerFaceColor", [0.49 0.83 0.99], "MarkerEdgeColor", "none", "MarkerFaceAlpha", 0.6);
    end
end

function drawCommittedObstacles(ax, rec)
    obs = getFieldOr(rec, "obstacles", []);
    if isempty(obs)
        return;
    end
    pts = zeros(numel(obs), 3);
    alphas = zeros(numel(obs), 1);
    for i = 1:numel(obs)
        pts(i, :) = [double(obs(i).x_m), double(obs(i).y_m), double(obs(i).z_m)];
        alphas(i) = double(getFieldOr(obs(i), "opacity", 0.25));
    end
    scatter3(ax, pts(:, 1), pts(:, 2), pts(:, 3), 120, ...
        "MarkerFaceColor", [0.55 0.24 0.82], ...
        "MarkerEdgeColor", "none", ...
        "MarkerFaceAlpha", max(0.18, mean(alphas)));
end

function drawObstacleGeometry(ax, rec)
    proj = getFieldOr(rec, "projection_debug", struct());
    channels = getFieldOr(proj, "channels", []);
    if isempty(channels)
        return;
    end

    for i = 1:numel(channels)
        ch = channels(i);
        cells = getFieldOr(ch, "cells", []);
        if isempty(cells)
            continue;
        end

        pts = [];
        for k = 1:numel(cells)
            cellEntry = cells(k);
            if ~logical(getFieldOr(cellEntry, "in_threshold", false)) || ~logical(getFieldOr(cellEntry, "feasible", false))
                continue;
            end
            pt = getFieldOr(cellEntry, "point_base_m", []);
            if numel(pt) >= 3
                pts(end + 1, :) = double(pt(1:3)); %#ok<AGROW>
            end
        end

        if isempty(pts)
            continue;
        end

        color = sensorColor(getFieldOr(ch, "channel", -1));
        scatter3(ax, pts(:, 1), pts(:, 2), pts(:, 3), 30, ...
            "MarkerFaceColor", color, ...
            "MarkerEdgeColor", "none", ...
            "MarkerFaceAlpha", 0.24);

        if size(pts, 1) >= 4
            try
                K = convhull(pts(:, 1), pts(:, 2), pts(:, 3));
                trisurf(K, pts(:, 1), pts(:, 2), pts(:, 3), ...
                    "Parent", ax, ...
                    "FaceColor", color, ...
                    "EdgeColor", color * 0.75, ...
                    "FaceAlpha", 0.08, ...
                    "EdgeAlpha", 0.12, ...
                    "LineWidth", 0.4);
            catch
                % Point cloud only when hull fails.
            end
        end
    end
end

function drawToFCoverage(ax, rec)
    proj = getFieldOr(rec, "projection_debug", struct());
    channels = getFieldOr(proj, "channels", []);
    for i = 1:numel(channels)
        ch = channels(i);
        mount = getFieldOr(ch, "sensor_mount", struct());
        origin = getFieldOr(mount, "origin_m", []);
        axisBase = getFieldOr(mount, "axis_base", []);
        rightBase = getFieldOr(mount, "right_base", []);
        upBase = getFieldOr(mount, "up_base", []);
        if numel(origin) < 3 || numel(axisBase) < 3 || numel(rightBase) < 3 || numel(upBase) < 3
            continue;
        end
        color = sensorColor(getFieldOr(ch, "channel", -1));
        coneRange = max(0.12, 1.15 * 0.20);
        [X, Y, Z] = conePatch(double(origin(1:3)), double(axisBase(1:3)), double(rightBase(1:3)), double(upBase(1:3)), coneRange, 63.0);
        surf(ax, X, Y, Z, "FaceColor", color, "FaceAlpha", 0.08, "EdgeAlpha", 0.05, "EdgeColor", color);
        plot3(ax, origin(1), origin(2), origin(3), "o", "MarkerFaceColor", color, "MarkerEdgeColor", "none", "MarkerSize", 5);
    end
end

function txt = buildStatusText(rec)
    obs = getFieldOr(rec, "obstacle", struct());
    plannerRoot = getFieldOr(rec, "planner_debug", struct());
    planner = getFieldOr(plannerRoot, "debug", struct());
    cmdVel = getFieldOr(rec, "cmd_velocity", struct());
    txt = sprintf([ ...
        'mode      : %s\n', ...
        'model     : %s\n', ...
        'obstacle  : %s  src=%s\n', ...
        'distance  : %.1f mm\n', ...
        'corridor  : %s\n', ...
        'severity  : %.2f\n', ...
        'clearance : %.1f mm\n', ...
        'speed     : %.2f\n', ...
        'theta vel : %.2f deg/s\n', ...
        'r vel     : %.2f mm/s\n', ...
        'z vel     : %.2f mm/s'], ...
        char(string(getFieldOr(rec, "mode", ""))), ...
        char(string(getFieldOr(rec, "obstacle_class", getFieldOr(plannerRoot, "obstacle_class", "POINT")))), ...
        char(string(getFieldOr(obs, "response", ""))), ...
        char(string(getFieldOr(obs, "source", ""))), ...
        double(getFieldOr(obs, "distance_mm", -1.0)), ...
        char(string(getFieldOr(planner, "corridor", ""))), ...
        double(getFieldOr(planner, "severity", 0.0)), ...
        double(getFieldOr(planner, "min_clearance_mm", nan)), ...
        double(getFieldOr(planner, "speed_scale", 1.0)), ...
        double(getFieldOr(cmdVel, "theta_deg_s", 0.0)), ...
        double(getFieldOr(cmdVel, "r_mm_s", 0.0)), ...
        double(getFieldOr(cmdVel, "z_mm_s", 0.0)));
end

function color = sensorColor(ch)
    switch double(ch)
        case 0
            color = [0.23 0.51 0.96];
        case 1
            color = [0.98 0.45 0.12];
        otherwise
            color = [0.65 0.35 0.85];
    end
end

function [X, Y, Z] = conePatch(origin, axisDir, rightDir, upDir, coneRange, fovDeg)
    axisDir = axisDir ./ max(norm(axisDir), 1e-9);
    rightDir = rightDir ./ max(norm(rightDir), 1e-9);
    upDir = upDir ./ max(norm(upDir), 1e-9);
    halfFov = deg2rad(fovDeg * 0.5);
    n = 22;
    ring = zeros(n, 3);
    for k = 1:n
        phi = 2.0 * pi * (k - 1) / n;
        ax = cos(phi) * halfFov;
        ay = sin(phi) * halfFov;
        d = axisDir + tan(ax) * rightDir + tan(ay) * upDir;
        d = d ./ max(norm(d), 1e-9);
        ring(k, :) = origin + coneRange * d;
    end
    X = [origin(1) * ones(1, n); ring(:, 1)'];
    Y = [origin(2) * ones(1, n); ring(:, 2)'];
    Z = [origin(3) * ones(1, n); ring(:, 3)'];
end

function arr = readNpyMatrix(path)
    arr = [];
    fid = fopen(path, "r");
    if fid < 0
        return;
    end
    cleanup = onCleanup(@() fclose(fid));
    magic = fread(fid, 6, "*char")';
    if ~strcmp(magic, char([147 'NUMPY']))
        return;
    end
    major = fread(fid, 1, "uint8"); %#ok<NASGU>
    minor = fread(fid, 1, "uint8"); %#ok<NASGU>
    headerLen = fread(fid, 1, "uint16");
    header = fread(fid, headerLen, "*char")';

    descr = regexp(header, "'descr':\s*'([^']+)'", "tokens", "once");
    shapeTok = regexp(header, "'shape':\s*\(([^)]*)\)", "tokens", "once");
    if isempty(descr) || isempty(shapeTok)
        return;
    end

    shapeVals = strtrim(split(string(shapeTok{1}), ","));
    shapeVals = shapeVals(shapeVals ~= "");
    dims = double(str2double(shapeVals));
    if any(isnan(dims))
        return;
    end

    switch descr{1}
        case "<f8"
            dtype = "double";
        case "<f4"
            dtype = "single";
        otherwise
            return;
    end

    raw = fread(fid, prod(dims), ["*" dtype]);
    if numel(raw) ~= prod(dims)
        return;
    end
    arr = reshape(double(raw), dims');
end
