import "./style.css";
import { Pose, POSE_CONNECTIONS } from "@mediapipe/pose";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";

const landmarkIndex = {
  nose: 0,
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
  leftHip: 23,
  rightHip: 24,
  leftKnee: 25,
  rightKnee: 26,
  leftAnkle: 27,
  rightAnkle: 28,
  leftHeel: 29,
  rightHeel: 30,
  leftFootIndex: 31,
  rightFootIndex: 32,
};

const elements = {
  video: document.querySelector("#video"),
  overlay: document.querySelector("#overlay"),
  videoStage: document.querySelector(".video-stage"),
  videoGuide: document.querySelector("#videoGuide"),
  videoGuideTitle: document.querySelector("#videoGuideTitle"),
  videoGuideText: document.querySelector("#videoGuideText"),
  cameraState: document.querySelector("#cameraState"),
  trackingState: document.querySelector("#trackingState"),
  exerciseName: document.querySelector("#exerciseName"),
  exerciseConfidence: document.querySelector("#exerciseConfidence"),
  primaryCue: document.querySelector("#primaryCue"),
  secondaryCue: document.querySelector("#secondaryCue"),
  repCount: document.querySelector("#repCount"),
  repStage: document.querySelector("#repStage"),
  holdTime: document.querySelector("#holdTime"),
  postureBadge: document.querySelector("#postureBadge"),
  elbowMetric: document.querySelector("#elbowMetric"),
  kneeMetric: document.querySelector("#kneeMetric"),
  torsoMetric: document.querySelector("#torsoMetric"),
  crunchMetric: document.querySelector("#crunchMetric"),
};

const canvasContext = elements.overlay.getContext("2d");

// Runtime memory for classification, rep counting, motion smoothing, and UI persistence.
const state = {
  currentExercise: "Unknown",
  candidateExercise: "Unknown",
  stableFrames: 0,
  poseTracked: false,
  repCount: 0,
  stage: "ready",
  holdStartedAt: null,
  holdSeconds: 0,
  exerciseScores: {
    pushups: 0,
    squats: 0,
    plank: 0,
    crunches: 0,
  },
  lastAverageElbow: null,
  elbowMotion: 0,
  lastHipY: null,
  hipMotion: 0,
  lastShoulderY: null,
  shoulderMotion: 0,
  pushupHighElbow: null,
  pushupLowElbow: null,
  pushupTopShoulderY: null,
  pushupBottomShoulderY: null,
  crunchOpenRatio: null,
  crunchClosedRatio: null,
  lastDisplaySnapshot: null,
  lastUpdateAt: performance.now(),
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function radiansToDegrees(radians) {
  return (radians * 180) / Math.PI;
}

function pointAverage(a, b) {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    z: ((a.z ?? 0) + (b.z ?? 0)) / 2,
    visibility: Math.min(a.visibility ?? 1, b.visibility ?? 1),
  };
}

function pointAverage3D(a, b) {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    z: ((a.z ?? 0) + (b.z ?? 0)) / 2,
  };
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function distance3D(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z ?? 0) - (b.z ?? 0));
}

function angleBetween(a, vertex, c) {
  const ab = { x: a.x - vertex.x, y: a.y - vertex.y };
  const cb = { x: c.x - vertex.x, y: c.y - vertex.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const magnitude = Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y);

  if (!magnitude) {
    return 0;
  }

  return radiansToDegrees(Math.acos(clamp(dot / magnitude, -1, 1)));
}

function angleBetween3D(a, vertex, c) {
  const ab = {
    x: a.x - vertex.x,
    y: a.y - vertex.y,
    z: (a.z ?? 0) - (vertex.z ?? 0),
  };
  const cb = {
    x: c.x - vertex.x,
    y: c.y - vertex.y,
    z: (c.z ?? 0) - (vertex.z ?? 0),
  };
  const dot = ab.x * cb.x + ab.y * cb.y + ab.z * cb.z;
  const magnitude =
    Math.hypot(ab.x, ab.y, ab.z) * Math.hypot(cb.x, cb.y, cb.z);

  if (!magnitude) {
    return 0;
  }

  return radiansToDegrees(Math.acos(clamp(dot / magnitude, -1, 1)));
}

function lineAngle(a, b) {
  const angle = Math.abs(radiansToDegrees(Math.atan2(b.y - a.y, b.x - a.x))) % 180;
  return angle > 90 ? 180 - angle : angle;
}

function getLandmark(landmarks, name) {
  return landmarks[landmarkIndex[name]];
}

function hasVisibility(point, min = 0.38) {
  return (point?.visibility ?? 0) >= min;
}

function meanAvailable(values, fallback = 0) {
  const valid = values.filter((value) => Number.isFinite(value));
  if (!valid.length) {
    return fallback;
  }

  return valid.reduce((sum, value) => sum + value, 0) / valid.length;
}

function safeAngle(a, vertex, c, min = 0.35) {
  if (![a, vertex, c].every((point) => hasVisibility(point, min))) {
    return NaN;
  }

  return angleBetween(a, vertex, c);
}

function safeAxisOffset(a, b, axis = "x", min = 0.35) {
  if (![a, b].every((point) => hasVisibility(point, min))) {
    return NaN;
  }

  return Math.abs(a[axis] - b[axis]);
}

function safeWorldAngle(a, vertex, c) {
  if (![a, vertex, c].every(Boolean)) {
    return NaN;
  }

  return angleBetween3D(a, vertex, c);
}

// Prefer the 3D body-space metric when available because it is less sensitive to mirror view
// and screen-space distortion than the 2D image metric.
function getEffectiveCrunchRatio(metrics) {
  return Number.isFinite(metrics.worldCrunchRatio) ? metrics.worldCrunchRatio : metrics.crunchRatio;
}

function getEffectiveStraightness(metrics) {
  return Number.isFinite(metrics.worldBodyStraightness)
    ? metrics.worldBodyStraightness
    : metrics.bodyStraightness;
}

// First-stage classifier: decide whether this frame looks like upright work, floor work,
// idle standing, or a transition pose before scoring specific exercises.
function classifyPoseFamily(metrics) {
  const floorAngle = Math.min(metrics.torsoAngle, metrics.bodyLineAngle);
  const floorPose =
    floorAngle < 56 && metrics.floorViewAngle < 46 && metrics.sideReliability > 0.24;
  const uprightPose =
    metrics.torsoAngle > 54 && metrics.averageHip < 176 && metrics.sideReliability > 0.24;
  const effectiveCrunchRatio = getEffectiveCrunchRatio(metrics);
  const effectiveStraightness = getEffectiveStraightness(metrics);
  const neutralStanding =
    uprightPose &&
    metrics.averageKnee > 160 &&
    metrics.averageHip > 162 &&
    metrics.hipMotion < 0.75;
  const supportedFloor =
    floorPose &&
    effectiveStraightness > 145 &&
    metrics.wristShoulderOffset < 0.58 &&
    metrics.averageHip > 138;
  const tuckedFloor =
    floorPose &&
    metrics.averageKnee < 144 &&
    metrics.averageHip < 162 &&
    effectiveCrunchRatio < 2.18;

  if (neutralStanding) {
    return {
      family: "idle",
      floorPose,
      uprightPose,
      neutralStanding,
      supportedFloor,
      tuckedFloor,
    };
  }

  if (supportedFloor || tuckedFloor || floorPose) {
    return {
      family: "floor",
      floorPose,
      uprightPose,
      neutralStanding,
      supportedFloor,
      tuckedFloor,
    };
  }

  if (uprightPose) {
    return {
      family: "upright",
      floorPose,
      uprightPose,
      neutralStanding,
      supportedFloor,
      tuckedFloor,
    };
  }

  return {
    family: "transition",
    floorPose,
    uprightPose,
    neutralStanding,
    supportedFloor,
    tuckedFloor,
  };
}

function getSideLandmarks(landmarks, side) {
  const prefix = side === "left" ? "left" : "right";

  return {
    shoulder: getLandmark(landmarks, `${prefix}Shoulder`),
    elbow: getLandmark(landmarks, `${prefix}Elbow`),
    wrist: getLandmark(landmarks, `${prefix}Wrist`),
    hip: getLandmark(landmarks, `${prefix}Hip`),
    knee: getLandmark(landmarks, `${prefix}Knee`),
    ankle: getLandmark(landmarks, `${prefix}Ankle`),
    foot: getLandmark(landmarks, `${prefix}FootIndex`),
  };
}

function sideVisibilityScore(sideLandmarks) {
  return meanAvailable(
    [
      sideLandmarks.shoulder?.visibility,
      sideLandmarks.elbow?.visibility,
      sideLandmarks.wrist?.visibility,
      sideLandmarks.hip?.visibility,
      sideLandmarks.knee?.visibility,
      sideLandmarks.ankle?.visibility,
    ],
    0,
  );
}

function preferVisiblePoint(primary, secondary, min = 0.35) {
  if (hasVisibility(primary, min) && hasVisibility(secondary, min)) {
    return pointAverage(primary, secondary);
  }

  if (hasVisibility(primary, min)) {
    return primary;
  }

  if (hasVisibility(secondary, min)) {
    return secondary;
  }

  return primary ?? secondary;
}

function isTracked(landmarks) {
  const left = getSideLandmarks(landmarks, "left");
  const right = getSideLandmarks(landmarks, "right");
  const leftSideTracked =
    hasVisibility(left.shoulder, 0.4) &&
    hasVisibility(left.hip, 0.36) &&
    hasVisibility(left.knee, 0.28) &&
    hasVisibility(left.ankle, 0.24);
  const rightSideTracked =
    hasVisibility(right.shoulder, 0.4) &&
    hasVisibility(right.hip, 0.36) &&
    hasVisibility(right.knee, 0.28) &&
    hasVisibility(right.ankle, 0.24);
  const coreTracked =
    (hasVisibility(left.shoulder, 0.4) || hasVisibility(right.shoulder, 0.4)) &&
    (hasVisibility(left.hip, 0.36) || hasVisibility(right.hip, 0.36)) &&
    (hasVisibility(left.knee, 0.28) || hasVisibility(right.knee, 0.28)) &&
    (hasVisibility(left.ankle, 0.24) || hasVisibility(right.ankle, 0.24));

  return leftSideTracked || rightSideTracked || coreTracked;
}

// Convert raw pose landmarks into the geometry signals used by the exercise judge.
function postureMetrics(landmarks, worldLandmarks = null) {
  const leftShoulder = getLandmark(landmarks, "leftShoulder");
  const rightShoulder = getLandmark(landmarks, "rightShoulder");
  const leftElbow = getLandmark(landmarks, "leftElbow");
  const rightElbow = getLandmark(landmarks, "rightElbow");
  const leftWrist = getLandmark(landmarks, "leftWrist");
  const rightWrist = getLandmark(landmarks, "rightWrist");
  const leftHip = getLandmark(landmarks, "leftHip");
  const rightHip = getLandmark(landmarks, "rightHip");
  const leftKnee = getLandmark(landmarks, "leftKnee");
  const rightKnee = getLandmark(landmarks, "rightKnee");
  const leftAnkle = getLandmark(landmarks, "leftAnkle");
  const rightAnkle = getLandmark(landmarks, "rightAnkle");
  const leftFoot = getLandmark(landmarks, "leftFootIndex");
  const rightFoot = getLandmark(landmarks, "rightFootIndex");

  const leftSide = getSideLandmarks(landmarks, "left");
  const rightSide = getSideLandmarks(landmarks, "right");
  const primarySide = sideVisibilityScore(leftSide) >= sideVisibilityScore(rightSide) ? "left" : "right";
  const dominant = primarySide === "left" ? leftSide : rightSide;

  const shoulderMid = preferVisiblePoint(leftShoulder, rightShoulder);
  const hipMid = preferVisiblePoint(leftHip, rightHip);
  const kneeMid = preferVisiblePoint(leftKnee, rightKnee);
  const ankleMid = preferVisiblePoint(leftAnkle, rightAnkle);

  const elbowLeft = safeAngle(leftShoulder, leftElbow, leftWrist);
  const elbowRight = safeAngle(rightShoulder, rightElbow, rightWrist);
  const kneeLeft = safeAngle(leftHip, leftKnee, leftAnkle);
  const kneeRight = safeAngle(rightHip, rightKnee, rightAnkle);
  const hipLeft = safeAngle(leftShoulder, leftHip, leftKnee);
  const hipRight = safeAngle(rightShoulder, rightHip, rightKnee);

  const averageElbow = meanAvailable(
    [elbowLeft, elbowRight],
    safeAngle(dominant.shoulder, dominant.elbow, dominant.wrist) || 160,
  );
  const averageKnee = meanAvailable(
    [kneeLeft, kneeRight],
    safeAngle(dominant.hip, dominant.knee, dominant.ankle) || 160,
  );
  const averageHip = meanAvailable(
    [hipLeft, hipRight],
    safeAngle(dominant.shoulder, dominant.hip, dominant.knee) || 160,
  );
  const torsoAngle = lineAngle(shoulderMid, hipMid);
  const bodyLineAngle = lineAngle(shoulderMid, ankleMid);
  const shoulderToKnee = distance(shoulderMid, kneeMid);
  const torsoLength = Math.max(distance(shoulderMid, hipMid), 0.0001);
  const crunchRatio = shoulderToKnee / torsoLength;
  const bodyStraightness = angleBetween(shoulderMid, hipMid, ankleMid);
  const wristShoulderOffset = meanAvailable(
    [safeAxisOffset(leftWrist, leftShoulder), safeAxisOffset(rightWrist, rightShoulder)],
    safeAxisOffset(dominant.wrist, dominant.shoulder) || 0,
  );
  const kneeFootDrift = meanAvailable(
    [safeAxisOffset(leftKnee, leftFoot), safeAxisOffset(rightKnee, rightFoot)],
    safeAxisOffset(dominant.knee, dominant.foot) || 0,
  );
  const hipHeightDelta = hipMid.y - (shoulderMid.y + ankleMid.y) / 2;
  const shoulderWidth =
    hasVisibility(leftShoulder, 0.35) && hasVisibility(rightShoulder, 0.35)
      ? Math.max(distance(leftShoulder, rightShoulder), 0.0001)
      : Math.max(distance(dominant.shoulder, dominant.hip) * 0.75, 0.0001);
  const ankleWidth =
    hasVisibility(leftAnkle, 0.35) && hasVisibility(rightAnkle, 0.35)
      ? Math.max(distance(leftAnkle, rightAnkle), 0.0001)
      : Math.max(distance(dominant.knee, dominant.ankle) * 0.9, 0.0001);
  const stanceRatio = ankleWidth / shoulderWidth;

  const worldLeftShoulder = worldLandmarks ? getLandmark(worldLandmarks, "leftShoulder") : null;
  const worldRightShoulder = worldLandmarks ? getLandmark(worldLandmarks, "rightShoulder") : null;
  const worldLeftHip = worldLandmarks ? getLandmark(worldLandmarks, "leftHip") : null;
  const worldRightHip = worldLandmarks ? getLandmark(worldLandmarks, "rightHip") : null;
  const worldLeftKnee = worldLandmarks ? getLandmark(worldLandmarks, "leftKnee") : null;
  const worldRightKnee = worldLandmarks ? getLandmark(worldLandmarks, "rightKnee") : null;
  const worldLeftAnkle = worldLandmarks ? getLandmark(worldLandmarks, "leftAnkle") : null;
  const worldRightAnkle = worldLandmarks ? getLandmark(worldLandmarks, "rightAnkle") : null;

  const worldShoulderMid =
    worldLeftShoulder && worldRightShoulder
      ? pointAverage3D(worldLeftShoulder, worldRightShoulder)
      : null;
  const worldHipMid =
    worldLeftHip && worldRightHip ? pointAverage3D(worldLeftHip, worldRightHip) : null;
  const worldKneeMid =
    worldLeftKnee && worldRightKnee ? pointAverage3D(worldLeftKnee, worldRightKnee) : null;
  const worldAnkleMid =
    worldLeftAnkle && worldRightAnkle ? pointAverage3D(worldLeftAnkle, worldRightAnkle) : null;

  const worldBodyStraightness = safeWorldAngle(worldShoulderMid, worldHipMid, worldAnkleMid);
  const worldCrunchRatio =
    worldShoulderMid && worldKneeMid && worldHipMid
      ? distance3D(worldShoulderMid, worldKneeMid) /
        Math.max(distance3D(worldShoulderMid, worldHipMid), 0.0001)
      : NaN;

  return {
    shoulderMid,
    hipMid,
    kneeMid,
    ankleMid,
    averageElbow,
    averageKnee,
    averageHip,
    torsoAngle,
    bodyLineAngle,
    crunchRatio,
    worldCrunchRatio,
    bodyStraightness,
    worldBodyStraightness,
    wristShoulderOffset,
    kneeFootDrift,
    hipHeightDelta,
    stanceRatio,
    primarySide,
    sideReliability: sideVisibilityScore(dominant),
    floorViewAngle: lineAngle(dominant.shoulder, dominant.ankle),
  };
}

// Second-stage classifier: score each supported exercise inside the currently detected pose family.
function scoreExercises(metrics, poseFamily) {
  const floorAngle = Math.min(metrics.torsoAngle, metrics.bodyLineAngle);
  const uprightScore = clamp((metrics.torsoAngle - 50) / 35, 0, 1);
  const horizontalScore = clamp((48 - floorAngle) / 48, 0, 1);
  const floorViewScore = clamp((36 - metrics.floorViewAngle) / 36, 0, 1);
  const squatDepth = clamp((160 - metrics.averageKnee) / 80, 0, 1);
  const pushDepth = clamp((165 - metrics.averageElbow) / 85, 0, 1);
  const effectiveCrunchRatio = getEffectiveCrunchRatio(metrics);
  const effectiveStraightness = getEffectiveStraightness(metrics);
  const crunchCurl = clamp((3.1 - effectiveCrunchRatio) / 1.4, 0, 1);
  const straightBody = clamp((180 - Math.abs(180 - effectiveStraightness)) / 180, 0, 1);
  const bentKneeScore = clamp((145 - metrics.averageKnee) / 55, 0, 1);
  const straightLegScore = clamp((metrics.averageKnee - 150) / 30, 0, 1);
  const bentHipScore = clamp((145 - metrics.averageHip) / 50, 0, 1);
  const straightHipScore = clamp((metrics.averageHip - 145) / 25, 0, 1);
  const elbowBentScore = clamp((155 - metrics.averageElbow) / 45, 0, 1);
  const flatHipScore = clamp((0.08 - Math.abs(metrics.hipHeightDelta)) / 0.08, 0, 1);
  const handsUnderShoulders = clamp((0.45 - metrics.wristShoulderOffset) / 0.45, 0, 1);
  const squatStanceScore = clamp((1.8 - Math.abs(metrics.stanceRatio - 1.4)) / 1.8, 0, 1);
  const crunchCompactness = clamp((2.35 - effectiveCrunchRatio) / 0.9, 0, 1);
  const floorReadiness = clamp(horizontalScore * 0.7 + floorViewScore * 0.5, 0, 1);
  const uprightReadiness = clamp(uprightScore * 0.8 + (1 - floorReadiness) * 0.35, 0, 1);
  const armMotionScore = clamp((metrics.elbowMotion - 2.5) / 10, 0, 1);
  const hipMotionScore = clamp((metrics.hipMotion - 0.55) / 2.4, 0, 1);
  const shoulderMotionScore = clamp((metrics.shoulderMotion - 0.8) / 3.2, 0, 1);
  const upperBodyMotionScore = Math.max(armMotionScore, shoulderMotionScore);
  const lockedArmScore = clamp((metrics.averageElbow - 150) / 20, 0, 1);
  const pushArmScore = Math.max(pushDepth, elbowBentScore * 0.8);
  const squatDepthReadiness = clamp(
    clamp((151 - metrics.averageKnee) / 20, 0, 1) * 0.58 +
      clamp((154 - metrics.averageHip) / 22, 0, 1) * 0.42,
    0,
    1,
  );
  const squatActivationScore = clamp(
    squatDepthReadiness * 0.72 + hipMotionScore * 0.28,
    0,
    1,
  );
  const neutralStandingPenalty =
    uprightScore *
    clamp((metrics.averageKnee - 154) / 18, 0, 1) *
    clamp((metrics.averageHip - 156) / 18, 0, 1) *
    clamp((1.1 - metrics.hipMotion) / 1.1, 0, 1);
  const squatPattern = clamp(
    squatDepth * 0.42 + bentHipScore * 0.23 + squatStanceScore * 0.2 + uprightScore * 0.15,
    0,
    1,
  );
  const pushSupportScore = clamp(
    handsUnderShoulders * 0.38 +
      straightBody * 0.24 +
      straightLegScore * 0.18 +
      straightHipScore * 0.1 +
      flatHipScore * 0.1,
    0,
    1,
  );
  const crunchTuckScore = clamp(
    crunchCompactness * 0.34 +
      bentKneeScore * 0.26 +
      bentHipScore * 0.22 +
      (1 - handsUnderShoulders) * 0.1 +
      (1 - straightLegScore) * 0.08,
    0,
    1,
  );
  const pushPattern = clamp(
    pushArmScore * 0.24 +
      straightBody * 0.24 +
      handsUnderShoulders * 0.16 +
      straightLegScore * 0.12 +
      straightHipScore * 0.12 +
      upperBodyMotionScore * 0.12,
    0,
    1,
  );
  const plankPattern = clamp(
    straightBody * 0.3 +
      flatHipScore * 0.28 +
      straightLegScore * 0.16 +
      lockedArmScore * 0.16 +
      floorViewScore * 0.1,
    0,
    1,
  );
  const crunchPattern = clamp(
    crunchCurl * 0.34 +
      crunchCompactness * 0.22 +
      bentHipScore * 0.2 +
      bentKneeScore * 0.18 +
      horizontalScore * 0.06,
    0,
    1,
  );

  const squatBoost =
    uprightScore > 0.52 && metrics.averageKnee < 154 && metrics.averageHip < 160 ? 0.12 : 0;
  const nonFloorPenalty = floorReadiness < 0.18 ? 0.18 : 0;
  const floorExerciseBoost = floorReadiness > 0.45 ? 0.08 : 0;
  const squatHorizontalPenalty = floorReadiness > 0.35 ? floorReadiness * 0.55 : 0;
  const pushupHorizontalBoost = floorReadiness > 0.35 ? floorReadiness * 0.12 : 0;
  const squatIdlePenalty = neutralStandingPenalty * 0.72;
  const squatActivationGate =
    state.currentExercise === "squats" ? 1 : clamp((squatActivationScore - 0.18) / 0.82, 0, 1);
  const pushupCrunchPenalty = crunchTuckScore * 0.34;
  const crunchSupportPenalty =
    pushSupportScore * 0.42 +
    straightBody * 0.14 +
    handsUnderShoulders * clamp((metrics.averageElbow - 148) / 20, 0, 1) * 0.35;
  const tabletopPose =
    poseFamily.family === "floor" &&
    metrics.averageKnee < 118 &&
    metrics.averageElbow > 145 &&
    handsUnderShoulders > 0.3 &&
    Math.abs(metrics.hipHeightDelta) > 0.035;
  const supportGate = poseFamily.supportedFloor
    ? clamp((pushSupportScore - 0.22) / 0.78, 0, 1)
    : poseFamily.family === "floor"
      ? 0.28
      : 0;
  const tuckGate = poseFamily.tuckedFloor
    ? clamp((crunchTuckScore - 0.24) / 0.76, 0, 1)
    : poseFamily.family === "floor"
      ? 0.22
      : 0;
  const floorGate = poseFamily.family === "floor" ? 1 : poseFamily.family === "transition" ? 0.45 : 0.08;
  const uprightGate = poseFamily.family === "upright" ? 1 : poseFamily.family === "transition" ? 0.45 : 0.05;

  if (poseFamily.family === "idle") {
    return {
      squats: 0,
      pushups: 0,
      plank: 0,
      crunches: 0,
    };
  }

  if (tabletopPose) {
    return {
      squats: 0,
      pushups: 0,
      plank: 0,
      crunches: 0,
    };
  }

  return {
    squats: clamp(
      (squatPattern * uprightReadiness + squatBoost - squatHorizontalPenalty - squatIdlePenalty) *
        squatActivationGate *
        uprightGate,
      0,
      1,
    ),
    pushups: clamp(
      pushPattern * floorReadiness -
        nonFloorPenalty +
        floorExerciseBoost +
        pushupHorizontalBoost -
        pushupCrunchPenalty,
      0,
      1,
    ) * supportGate * floorGate * clamp(0.72 + upperBodyMotionScore * 0.68, 0, 1.3),
    plank: clamp(
      plankPattern * floorReadiness - nonFloorPenalty - upperBodyMotionScore * 0.34 + floorExerciseBoost,
      0,
      1,
    ) * clamp(Math.max(supportGate, lockedArmScore), 0, 1) * floorGate,
    crunches: clamp(
      crunchPattern * horizontalScore * 1.1 - nonFloorPenalty - crunchSupportPenalty,
      0,
      1,
    ) * Math.max(tuckGate, poseFamily.family === "floor" ? 0.18 : 0) * floorGate,
  };
}

// Pick the winning exercise after scoring, then apply family sanity checks, readiness gates,
// tie-breaks, and hysteresis so the label does not flicker between nearby classes.
function chooseExercise(scores, metrics, poseFamily) {
  const entries = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const [name, confidence] = entries[0];
  const [, runnerUp = 0] = entries[1] ?? [];
  const currentScore = state.currentExercise !== "Unknown" ? scores[state.currentExercise] ?? 0 : 0;
  const currentIsFloorExercise =
    state.currentExercise === "pushups" ||
    state.currentExercise === "plank" ||
    state.currentExercise === "crunches";
  const currentIsSquat = state.currentExercise === "squats";
  const squatActivationReady =
    (metrics.averageKnee < 148 && metrics.averageHip < 160) ||
    (metrics.averageKnee < 152 && metrics.averageHip < 164 && metrics.hipMotion > 1.1);
  const floorPose = poseFamily.floorPose;
  const uprightPose = poseFamily.uprightPose;

  if (poseFamily.family === "idle") {
    return { name: "Unknown", confidence: 0 };
  }

  if (
    state.currentExercise !== "Unknown" &&
    !(currentIsSquat && floorPose) &&
    !(currentIsFloorExercise && uprightPose) &&
    currentScore >= 0.15 &&
    currentScore >= confidence - 0.05
  ) {
    return { name: state.currentExercise, confidence: currentScore };
  }

  if (!name || confidence < 0.13) {
    return { name: "Unknown", confidence: 0 };
  }

  if (name === "squats" && state.currentExercise !== "squats" && !squatActivationReady) {
    return { name: "Unknown", confidence: 0 };
  }

  const pushupReady =
    poseFamily.family === "floor" &&
    metrics.averageHip > 128 &&
    metrics.torsoAngle < 40 &&
    metrics.wristShoulderOffset < 0.62;
  const pushupMotionReady =
    pushupReady &&
    Math.max(metrics.elbowMotion, metrics.shoulderMotion) > 0.8 &&
    (metrics.averageElbow < 166 || metrics.shoulderMotion > 1.0);
  const crunchReady =
    poseFamily.family === "floor" &&
    metrics.averageKnee < 144 &&
    metrics.averageHip < 162 &&
    getEffectiveCrunchRatio(metrics) < 2.18 &&
    !(metrics.averageElbow > 148 && metrics.wristShoulderOffset < 0.34);
  const plankReady =
    poseFamily.family === "floor" &&
    metrics.averageElbow > 145 &&
    metrics.hipMotion < 1.2 &&
    Math.abs(metrics.hipHeightDelta) < 0.1;
  const tabletopPose =
    poseFamily.family === "floor" &&
    metrics.averageKnee < 118 &&
    metrics.averageElbow > 145 &&
    metrics.wristShoulderOffset < 0.36 &&
    Math.abs(metrics.hipHeightDelta) > 0.035;

  if (tabletopPose) {
    return { name: "Unknown", confidence: 0 };
  }

  if (poseFamily.family === "upright" && name !== "squats") {
    return { name: "Unknown", confidence: 0 };
  }

  if (poseFamily.family === "floor" && name === "squats") {
    return { name: "Unknown", confidence: 0 };
  }

  if (pushupMotionReady && (scores.pushups ?? 0) >= (scores.plank ?? 0) - 0.24) {
    return { name: "pushups", confidence: Math.max(scores.pushups ?? confidence, confidence) };
  }

  if (plankReady && (scores.plank ?? 0) >= confidence - 0.1) {
    return { name: "plank", confidence: scores.plank ?? confidence };
  }

  if (name === "crunches" && pushupReady && (scores.pushups ?? 0) >= confidence - 0.12) {
    return { name: "pushups", confidence: scores.pushups ?? confidence };
  }

  if (name === "pushups" && crunchReady && (scores.crunches ?? 0) >= confidence - 0.12) {
    return { name: "crunches", confidence: scores.crunches ?? confidence };
  }

  if (
    confidence - runnerUp < 0.02 &&
    state.currentExercise !== "Unknown" &&
    !(currentIsSquat && floorPose) &&
    !(currentIsFloorExercise && uprightPose)
  ) {
    return { name: state.currentExercise, confidence: currentScore };
  }

  return { name, confidence };
}

function humanizeExercise(name) {
  if (name === "pushups") return "Push-ups";
  if (name === "squats") return "Squats";
  if (name === "plank") return "Plank";
  if (name === "crunches") return "Crunches";
  return "Unknown";
}

function resetForExercise(exercise) {
  state.currentExercise = exercise;
  state.repCount = 0;
  state.stage = "ready";
  state.holdStartedAt = null;
  state.holdSeconds = 0;
  state.pushupHighElbow = null;
  state.pushupLowElbow = null;
  state.pushupTopShoulderY = null;
  state.pushupBottomShoulderY = null;
  state.crunchOpenRatio = null;
  state.crunchClosedRatio = null;
}

// Require the same candidate exercise to appear for multiple frames before switching.
function updateExerciseStability(nextExercise) {
  if (nextExercise === "Unknown") {
    state.candidateExercise = "Unknown";
    state.stableFrames = 0;
    return;
  }

  if (nextExercise === state.candidateExercise) {
    state.stableFrames += 1;
  } else {
    state.candidateExercise = nextExercise;
    state.stableFrames = 1;
  }

  if (nextExercise !== state.currentExercise && state.stableFrames >= 3) {
    resetForExercise(nextExercise);
  }
}

// Exercise-specific repetition and hold-time rules once an exercise has been confirmed.
function handleRepCounting(metrics, now) {
  if (state.currentExercise === "squats") {
    if (metrics.averageKnee < 95) {
      state.stage = "down";
    } else if (metrics.averageKnee > 160 && state.stage === "down") {
      state.stage = "up";
      state.repCount += 1;
    }
  }

  if (state.currentExercise === "pushups") {
    const elbow = metrics.averageElbow;
    const shoulderY = metrics.shoulderMid.y;
    if (state.stage !== "down") {
      state.pushupHighElbow =
        state.pushupHighElbow == null ? elbow : Math.max(state.pushupHighElbow, elbow);
      state.pushupTopShoulderY =
        state.pushupTopShoulderY == null ? shoulderY : Math.min(state.pushupTopShoulderY, shoulderY);
    }

    const highElbow = state.pushupHighElbow ?? elbow;
    const pushupDownThreshold = Math.min(138, highElbow - 10);
    const topShoulderY = state.pushupTopShoulderY ?? shoulderY;
    const shoulderDownThreshold = topShoulderY + 0.016;

    if (
      (elbow < pushupDownThreshold && highElbow - elbow > 8) ||
      shoulderY > shoulderDownThreshold
    ) {
      state.stage = "down";
      state.pushupLowElbow =
        state.pushupLowElbow == null ? elbow : Math.min(state.pushupLowElbow, elbow);
      state.pushupBottomShoulderY =
        state.pushupBottomShoulderY == null
          ? shoulderY
          : Math.max(state.pushupBottomShoulderY, shoulderY);
    } else if (state.stage === "down") {
      state.pushupLowElbow =
        state.pushupLowElbow == null ? elbow : Math.min(state.pushupLowElbow, elbow);
      state.pushupBottomShoulderY =
        state.pushupBottomShoulderY == null
          ? shoulderY
          : Math.max(state.pushupBottomShoulderY, shoulderY);
      const lowElbow = state.pushupLowElbow ?? elbow;
      const pushupUpThreshold = Math.max(136, lowElbow + 12);
      const bottomShoulderY = state.pushupBottomShoulderY ?? shoulderY;
      const shoulderUpReady =
        shoulderY < bottomShoulderY - 0.012 && shoulderY <= topShoulderY + 0.012;

      if (elbow > pushupUpThreshold || shoulderUpReady) {
        state.stage = "up";
        state.repCount += 1;
        state.pushupHighElbow = elbow;
        state.pushupLowElbow = null;
        state.pushupTopShoulderY = shoulderY;
        state.pushupBottomShoulderY = null;
      }
    } else if (elbow > highElbow) {
      state.stage = "up";
      state.pushupTopShoulderY =
        state.pushupTopShoulderY == null ? shoulderY : Math.min(state.pushupTopShoulderY, shoulderY);
    }
  }

  if (state.currentExercise === "crunches") {
    const crunchRatio = getEffectiveCrunchRatio(metrics);
    if (state.stage !== "up") {
      state.crunchOpenRatio =
        state.crunchOpenRatio == null ? crunchRatio : Math.max(state.crunchOpenRatio, crunchRatio);
    }

    const crunchUpThreshold = 1.5;
    const openRatio = state.crunchOpenRatio ?? crunchRatio;
    const crunchReleaseThreshold = Math.max(1.62, crunchUpThreshold + 0.1);
    const hadOpenStart = openRatio >= crunchUpThreshold + 0.08;

    if (crunchRatio <= crunchUpThreshold) {
      const enteringClosedPosition = state.stage !== "up";
      state.stage = "up";
      state.crunchClosedRatio =
        state.crunchClosedRatio == null
          ? crunchRatio
          : Math.min(state.crunchClosedRatio, crunchRatio);

      if (enteringClosedPosition && hadOpenStart) {
        state.repCount += 1;
      }
    } else if (state.stage === "up") {
      if (crunchRatio > crunchReleaseThreshold) {
        state.stage = "down";
        state.crunchOpenRatio = crunchRatio;
        state.crunchClosedRatio = null;
      }
    } else {
      state.stage = "down";
    }
  }

  if (state.currentExercise === "plank") {
    if (!state.holdStartedAt) {
      state.holdStartedAt = now;
    }

    state.holdSeconds = Math.max(0, (now - state.holdStartedAt) / 1000);
  } else {
    state.holdStartedAt = null;
    state.holdSeconds = 0;
  }
}

function formatDuration(seconds) {
  const totalSeconds = Math.floor(seconds);
  const mins = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
  const secs = String(totalSeconds % 60).padStart(2, "0");
  return `${mins}:${secs}`;
}

function formatAngleValue(value) {
  return `${Math.round(value)}°`;
}

function getExercisePostureScore(metrics) {
  if (state.currentExercise === "squats") {
    const depthScore = clamp((160 - metrics.averageKnee) / 45, 0, 1);
    const chestScore = clamp((metrics.torsoAngle - 58) / 18, 0, 1);
    const trackingScore = clamp((0.2 - metrics.kneeFootDrift) / 0.2, 0, 1);
    return Math.round((depthScore * 0.45 + chestScore * 0.3 + trackingScore * 0.25) * 100);
  }

  if (state.currentExercise === "pushups") {
    const lineScore = clamp((0.09 - Math.abs(metrics.hipHeightDelta)) / 0.09, 0, 1);
    const depthScore = clamp((160 - metrics.averageElbow) / 55, 0, 1);
    const handScore = clamp((0.62 - metrics.wristShoulderOffset) / 0.62, 0, 1);
    return Math.round((lineScore * 0.4 + depthScore * 0.35 + handScore * 0.25) * 100);
  }

  if (state.currentExercise === "plank") {
    const lineScore = clamp((0.08 - Math.abs(metrics.hipHeightDelta)) / 0.08, 0, 1);
    const armScore = clamp((metrics.averageElbow - 145) / 20, 0, 1);
    const straightScore = clamp((getEffectiveStraightness(metrics) - 145) / 25, 0, 1);
    return Math.round((lineScore * 0.45 + armScore * 0.2 + straightScore * 0.35) * 100);
  }

  if (state.currentExercise === "crunches") {
    const curlScore = clamp((2.35 - getEffectiveCrunchRatio(metrics)) / 0.75, 0, 1);
    const kneeScore = clamp((145 - metrics.averageKnee) / 25, 0, 1);
    const controlScore = clamp((metrics.hipMotion - 0.2) / 1.6, 0, 1);
    return Math.round((curlScore * 0.45 + kneeScore * 0.25 + controlScore * 0.3) * 100);
  }

  return Math.round(clamp(getEffectiveStraightness(metrics) / 180, 0, 1) * 100);
}

// Generate live coaching text from the current exercise plus the measured body geometry.
function buildFeedback(metrics) {
  if (state.currentExercise === "squats") {
    if (metrics.averageKnee > 125) {
      return {
        primary: "Sit deeper for a full squat.",
        secondary: `Knee angle is ${formatAngleValue(metrics.averageKnee)}. Aim lower before standing back up.`,
        tone: "warn",
      };
    }

    if (metrics.torsoAngle < 62) {
      return {
        primary: "Lift your chest more.",
        secondary: `Torso angle is ${formatAngleValue(metrics.torsoAngle)}. Keep your chest taller to stay balanced.`,
        tone: "warn",
      };
    }

    if (metrics.kneeFootDrift > 0.18) {
      return {
        primary: "Keep your knees tracking over your feet.",
        secondary: `Your knees are drifting forward. Push the hips back and keep the weight through mid-foot.`,
        tone: "warn",
      };
    }

    return {
      primary: "Strong squat form.",
      secondary: `Depth ${formatAngleValue(metrics.averageKnee)}, torso ${formatAngleValue(metrics.torsoAngle)}. Keep the pace steady.`,
      tone: "good",
    };
  }

  if (state.currentExercise === "pushups") {
    if (Math.abs(metrics.hipHeightDelta) > 0.055) {
      return {
        primary:
          metrics.hipHeightDelta > 0 ? "Raise your hips slightly." : "Lower your hips slightly.",
        secondary: `Body line is breaking at the hips. Keep shoulders, hips, and ankles in one long line.`,
        tone: "warn",
      };
    }

    if (metrics.averageElbow > 110 && state.stage === "down") {
      return {
        primary: "Lower your chest more.",
        secondary: `Elbow angle is ${formatAngleValue(metrics.averageElbow)}. Bend more before driving back up.`,
        tone: "warn",
      };
    }

    if (metrics.wristShoulderOffset > 0.6) {
      return {
        primary: "Stack your hands under your shoulders.",
        secondary: "Bring the hands a bit closer under the shoulder line for better stability and depth.",
        tone: "warn",
      };
    }

    return {
      primary: "Push-up line looks solid.",
      secondary: `Elbows ${formatAngleValue(metrics.averageElbow)}, hips steady. Keep bending and extending smoothly.`,
      tone: "good",
    };
  }

  if (state.currentExercise === "plank") {
    if (metrics.hipHeightDelta > 0.05) {
      return {
        primary: "Raise your hips a little.",
        secondary: "Your hips are sagging. Squeeze the core and glutes to lift into one straight line.",
        tone: "warn",
      };
    }

    if (metrics.hipHeightDelta < -0.05) {
      return {
        primary: "Lower your hips slightly.",
        secondary: "Your hips are too high. Lower them until shoulders, hips, and ankles line up.",
        tone: "warn",
      };
    }

    if (metrics.averageElbow < 150) {
      return {
        primary: "Straighten your arms more.",
        secondary: `Elbow angle is ${formatAngleValue(metrics.averageElbow)}. Straighter arms will make the hold more stable.`,
        tone: "warn",
      };
    }

    return {
      primary: "Plank is well aligned.",
      secondary: `Body line and hips look stable. Hold steady and keep breathing with a tight core.`,
      tone: "good",
    };
  }

  if (state.currentExercise === "crunches") {
    const effectiveCrunchRatio = getEffectiveCrunchRatio(metrics);

    if (effectiveCrunchRatio > 2.2 && state.stage === "up") {
      return {
        primary: "Curl your shoulders higher.",
        secondary: `Crunch ratio is ${effectiveCrunchRatio.toFixed(2)}. Think ribs toward hips instead of pulling with the neck.`,
        tone: "warn",
      };
    }

    if (metrics.averageKnee > 135) {
      return {
        primary: "Keep your knees more bent.",
        secondary: `Knee angle is ${formatAngleValue(metrics.averageKnee)}. Plant the feet and bend the knees more.`,
        tone: "warn",
      };
    }

    return {
      primary: "Crunch pattern looks controlled.",
      secondary: `Crunch ratio ${effectiveCrunchRatio.toFixed(2)}. Lift with your core and lower back down under control.`,
      tone: "good",
    };
  }

  return {
    primary: "Body detected. Start a clear exercise shape.",
    secondary: "Stand tall for squats. Turn sideways for push-ups, plank, and crunches.",
    tone: "warn",
  };
}

// When tracking is lost, keep the feedback specific to the last active exercise so the user
// knows what to fix without losing context.
function buildTrackingPausedFeedback() {
  if (state.currentExercise === "pushups" || state.currentExercise === "plank") {
    return {
      primary: "Tracking paused. Keep your full side view visible.",
      secondary: "Step back until shoulders, hips, knees, and ankles return to frame.",
      tone: "warn",
    };
  }

  if (state.currentExercise === "squats") {
    return {
      primary: "Tracking paused. Bring your full body back into frame.",
      secondary: "We need knees and ankles visible again before the next rep can be judged.",
      tone: "warn",
    };
  }

  if (state.currentExercise === "crunches") {
    return {
      primary: "Tracking paused. Re-center your body in frame.",
      secondary: "Keep knees, hips, and shoulders visible so the crunch pattern stays locked.",
      tone: "warn",
    };
  }

  return {
    primary: "Tracking paused. Step back until your full body fits.",
    secondary: "We need shoulders, hips, knees, and ankles visible before counting resumes.",
    tone: "warn",
  };
}

// Reuse framing guidance as main feedback whenever classification is intentionally blocked.
function buildRepositionFeedback(guide) {
  return {
    primary: guide.title,
    secondary: guide.text,
    tone: "warn",
  };
}

function setPillState(element, text, type = "neutral") {
  element.textContent = text;
  element.className = `pill ${type === "good" ? "pill-good" : type === "warn" ? "pill-warn" : ""}`.trim();
}

function setVideoGuide(title, text) {
  elements.videoGuideTitle.textContent = title;
  elements.videoGuideText.textContent = text;
}

function getVisibleBodyPoints(landmarks) {
  return [
    "leftShoulder",
    "rightShoulder",
    "leftHip",
    "rightHip",
    "leftKnee",
    "rightKnee",
    "leftAnkle",
    "rightAnkle",
    "leftElbow",
    "rightElbow",
    "leftWrist",
    "rightWrist",
  ]
    .map((name) => getLandmark(landmarks, name))
    .filter((point) => hasVisibility(point, 0.35));
}

function positionVideoGuide() {
  elements.videoGuide.classList.remove("is-floating");
  elements.videoGuide.style.left = "50%";
  elements.videoGuide.style.bottom = "24px";
  elements.videoGuide.style.removeProperty("top");
}

// Camera/framing gate: if the body is too close, too far, cropped, dim, or missing key joints,
// pause classification instead of risking a wrong exercise label.
function assessFraming(landmarks) {
  if (!landmarks) {
    return {
      title: "Step back so your full body fits.",
      text: "We need shoulders, hips, knees, and ankles visible before counting starts.",
      visible: true,
      classificationBlocked: true,
    };
  }

  const leftKnee = getLandmark(landmarks, "leftKnee");
  const rightKnee = getLandmark(landmarks, "rightKnee");
  const leftAnkle = getLandmark(landmarks, "leftAnkle");
  const rightAnkle = getLandmark(landmarks, "rightAnkle");
  const kneeVisible = hasVisibility(leftKnee, 0.26) || hasVisibility(rightKnee, 0.26);
  const ankleVisible = hasVisibility(leftAnkle, 0.22) || hasVisibility(rightAnkle, 0.22);

  if (!kneeVisible) {
    return {
      title: "Step back until your knees are visible.",
      text: "The coach cannot count or score form well until at least one knee is clearly in frame.",
      visible: true,
      classificationBlocked: true,
    };
  }

  if (!ankleVisible) {
    return {
      title: "Step back a bit more so ankles show too.",
      text: "Knees are visible now. We also want ankles in frame for steadier lower-body tracking.",
      visible: true,
      classificationBlocked: true,
    };
  }

  const points = getVisibleBodyPoints(landmarks);
  if (points.length < 5) {
    return {
      title: "Move into brighter, clearer framing.",
      text: "Stand where your shoulders, hips, knees, and ankles are easier to see.",
      visible: true,
      classificationBlocked: true,
    };
  }

  const minX = Math.min(...points.map((point) => point.x));
  const maxX = Math.max(...points.map((point) => point.x));
  const minY = Math.min(...points.map((point) => point.y));
  const maxY = Math.max(...points.map((point) => point.y));
  const bodyHeight = maxY - minY;
  const bodyWidth = maxX - minX;
  const centerX = (minX + maxX) / 2;
  const framePaddingY = Math.min(minY, 1 - maxY);
  const framePaddingX = Math.min(minX, 1 - maxX);

  if (
    bodyHeight > 0.92 ||
    bodyWidth > 0.84 ||
    framePaddingY < 0.02 ||
    framePaddingX < 0.015
  ) {
    return {
      title: "Step back for a cleaner full-body view.",
      text: "You are too close right now. We would rather pause than guess the wrong exercise.",
      visible: true,
      classificationBlocked: true,
    };
  }

  if (bodyHeight < 0.26) {
    return {
      title: "Move a little closer, then hold still.",
      text: "We need a bit more body detail before exercise detection can be trusted.",
      visible: true,
      classificationBlocked: true,
    };
  }

  if (centerX < 0.24 || centerX > 0.76) {
    return {
      title: "Move back toward the center.",
      text: "Try to keep your body centered so the landmark points stay stable.",
      visible: true,
      classificationBlocked: false,
    };
  }

  if (state.currentExercise === "Unknown") {
    return {
      title: "Good framing. Hold still for a moment.",
      text: "For push-ups and plank, a side or slight three-quarter view usually works best.",
      visible: false,
      classificationBlocked: false,
    };
  }

  if (state.currentExercise === "pushups" || state.currentExercise === "plank") {
    return {
      title: "Strong view for floor work.",
      text: "Keep one shoulder-to-ankle side clearly visible while you move.",
      visible: false,
      classificationBlocked: false,
    };
  }

  return {
    title: "You are framed well.",
    text: "Stay inside the box and keep moving smoothly for the most stable tracking.",
    visible: false,
    classificationBlocked: false,
  };
}

function getVideoGuideMessage(landmarks) {
  const { title, text, visible } = assessFraming(landmarks);
  return { title, text, visible };
}

// Preserve the last valid dashboard reading so short tracking dropouts do not blank the UI.
function saveCurrentDisplaySnapshot() {
  state.lastDisplaySnapshot = {
    exerciseName: elements.exerciseName.textContent,
    exerciseConfidence: elements.exerciseConfidence.textContent,
    repCount: elements.repCount.textContent,
    repStage: elements.repStage.textContent,
    holdTime: elements.holdTime.textContent,
    postureBadge: elements.postureBadge.textContent,
    elbowMetric: elements.elbowMetric.textContent,
    kneeMetric: elements.kneeMetric.textContent,
    torsoMetric: elements.torsoMetric.textContent,
    crunchMetric: elements.crunchMetric.textContent,
  };
}

function renderLastSnapshot() {
  if (!state.lastDisplaySnapshot) {
    return false;
  }

  const snapshot = state.lastDisplaySnapshot;
  elements.exerciseName.textContent = snapshot.exerciseName;
  elements.exerciseConfidence.textContent = snapshot.exerciseConfidence;
  elements.repCount.textContent = snapshot.repCount;
  elements.repStage.textContent = snapshot.repStage;
  elements.holdTime.textContent = snapshot.holdTime;
  elements.postureBadge.textContent = snapshot.postureBadge;
  elements.elbowMetric.textContent = snapshot.elbowMetric;
  elements.kneeMetric.textContent = snapshot.kneeMetric;
  elements.torsoMetric.textContent = snapshot.torsoMetric;
  elements.crunchMetric.textContent = snapshot.crunchMetric;
  return true;
}

function renderMetrics(metrics, confidence) {
  const effectiveCrunchRatio = getEffectiveCrunchRatio(metrics);

  elements.exerciseName.textContent = humanizeExercise(state.currentExercise);
  elements.exerciseConfidence.textContent = `Confidence: ${Math.round(confidence * 100)}%`;
  elements.repCount.textContent = String(state.repCount);
  elements.repStage.textContent = `Stage: ${state.stage}`;
  elements.holdTime.textContent = formatDuration(state.holdSeconds);
  elements.postureBadge.textContent = `Form score: ${getExercisePostureScore(metrics)}%`;
  elements.elbowMetric.textContent = `${Math.round(metrics.averageElbow)}°`;
  elements.kneeMetric.textContent = `${Math.round(metrics.averageKnee)}°`;
  elements.torsoMetric.textContent = `${Math.round(metrics.torsoAngle)}°`;
  elements.crunchMetric.textContent = effectiveCrunchRatio.toFixed(2);
}

// Feedback and status pills are derived from the current tracking state plus the judge output.
function renderFeedback(feedback) {
  elements.primaryCue.textContent = feedback.primary;
  elements.secondaryCue.textContent = feedback.secondary;
  setPillState(
    elements.trackingState,
    !state.poseTracked
      ? "Body not fully visible"
      : state.currentExercise === "Unknown"
        ? "Body detected"
        : "Pose tracking active",
    !state.poseTracked ? "warn" : state.currentExercise === "Unknown" ? "neutral" : feedback.tone,
  );
}

function renderVideoGuide(guide) {
  positionVideoGuide();
  setVideoGuide(guide.title, guide.text);
  elements.videoGuide.classList.toggle("is-hidden", !guide.visible);
}

// Keep the drawing canvas aligned with the displayed video size at the current device pixel ratio.
function resizeCanvas() {
  const rect = elements.video.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  elements.overlay.width = rect.width * ratio;
  elements.overlay.height = rect.height * ratio;
  canvasContext.setTransform(ratio, 0, 0, ratio, 0, 0);
}

// Draw the current skeleton on top of the mirrored webcam feed.
function drawPose(landmarks) {
  const width = elements.overlay.clientWidth;
  const height = elements.overlay.clientHeight;

  canvasContext.clearRect(0, 0, width, height);
  drawConnectors(canvasContext, landmarks, POSE_CONNECTIONS, {
    color: "rgba(255, 196, 94, 0.85)",
    lineWidth: 3,
  });
  drawLandmarks(canvasContext, landmarks, {
    color: "rgba(255, 247, 224, 0.95)",
    fillColor: "rgba(255, 113, 67, 0.95)",
    radius: 4,
  });
}

// Main judge loop for each MediaPipe result: validate tracking, block bad framing, compute
// metrics, score exercises, confirm the winner, count reps, and update the UI.
function processResults(results) {
  const now = performance.now();
  const landmarks = results.poseLandmarks;

  if (!landmarks || !isTracked(landmarks)) {
    state.poseTracked = false;
    state.candidateExercise = "Unknown";
    state.stableFrames = 0;
    state.holdStartedAt = null;
    state.lastAverageElbow = null;
    state.elbowMotion = 0;
    state.lastHipY = null;
    state.hipMotion = 0;
    state.lastShoulderY = null;
    state.shoulderMotion = 0;
    renderVideoGuide(
      landmarks
        ? getVideoGuideMessage(landmarks)
        : {
            title: "Step back so your full body fits.",
            text: "We need shoulders, hips, knees, and ankles visible before counting starts.",
            visible: true,
          },
    );
    renderFeedback(buildTrackingPausedFeedback());
    renderLastSnapshot();
    canvasContext.clearRect(0, 0, elements.overlay.clientWidth, elements.overlay.clientHeight);
    return;
  }

  // Bad framing is treated differently from missing landmarks: keep the last stats visible,
  // but pause classification until distance and alignment are trustworthy again.
  const framing = assessFraming(landmarks);
  if (framing.classificationBlocked) {
    state.poseTracked = false;
    state.candidateExercise = "Unknown";
    state.stableFrames = 0;
    state.holdStartedAt = null;
    state.lastAverageElbow = null;
    state.elbowMotion = 0;
    state.lastHipY = null;
    state.hipMotion = 0;
    state.lastShoulderY = null;
    state.shoulderMotion = 0;
    renderVideoGuide(framing);
    renderFeedback(buildRepositionFeedback(framing));
    renderLastSnapshot();
    canvasContext.clearRect(0, 0, elements.overlay.clientWidth, elements.overlay.clientHeight);
    return;
  }

  state.poseTracked = true;
  renderVideoGuide(framing);
  drawPose(landmarks);

  // Build per-frame posture metrics and smooth motion signals over time instead of using
  // single-frame deltas directly.
  const metrics = postureMetrics(landmarks, results.poseWorldLandmarks ?? null);
  if (Number.isFinite(state.lastAverageElbow)) {
    const elbowDelta = Math.abs(metrics.averageElbow - state.lastAverageElbow);
    state.elbowMotion = state.elbowMotion * 0.7 + elbowDelta * 0.3;
  } else {
    state.elbowMotion = 0;
  }
  state.lastAverageElbow = metrics.averageElbow;
  if (Number.isFinite(state.lastHipY)) {
    const hipDelta = Math.abs(metrics.hipMid.y - state.lastHipY) * 100;
    state.hipMotion = state.hipMotion * 0.7 + hipDelta * 0.3;
  } else {
    state.hipMotion = 0;
  }
  state.lastHipY = metrics.hipMid.y;
  if (Number.isFinite(state.lastShoulderY)) {
    const shoulderDelta = Math.abs(metrics.shoulderMid.y - state.lastShoulderY) * 100;
    state.shoulderMotion = state.shoulderMotion * 0.7 + shoulderDelta * 0.3;
  } else {
    state.shoulderMotion = 0;
  }
  state.lastShoulderY = metrics.shoulderMid.y;
  metrics.elbowMotion = state.elbowMotion;
  metrics.hipMotion = state.hipMotion;
  metrics.shoulderMotion = state.shoulderMotion;

  // Judge in two phases: pose family first, then exercise scoring inside that family.
  const poseFamily = classifyPoseFamily(metrics);
  const scores = scoreExercises(metrics, poseFamily);

  // Decay impossible class families before smoothing the new frame into the running scores.
  if (poseFamily.family === "floor") {
    state.exerciseScores.squats *= 0.18;
  } else if (poseFamily.family === "upright") {
    state.exerciseScores.pushups *= 0.3;
    state.exerciseScores.plank *= 0.3;
    state.exerciseScores.crunches *= 0.3;
  } else if (poseFamily.family === "idle") {
    state.exerciseScores.squats *= 0.2;
    state.exerciseScores.pushups *= 0.2;
    state.exerciseScores.plank *= 0.2;
    state.exerciseScores.crunches *= 0.2;
  }

  for (const [name, score] of Object.entries(scores)) {
    state.exerciseScores[name] = state.exerciseScores[name] * 0.5 + score * 0.5;
  }

  // Confirm the best exercise, then run counting and feedback using the confirmed label.
  const choice = chooseExercise(state.exerciseScores, metrics, poseFamily);
  updateExerciseStability(choice.name);
  handleRepCounting(metrics, now);
  renderMetrics(metrics, choice.confidence);
  renderFeedback(buildFeedback(metrics));
  saveCurrentDisplaySnapshot();
}

async function waitForVideoReady(video) {
  if (video.readyState >= 2) {
    return;
  }

  await new Promise((resolve) => {
    video.onloadeddata = () => resolve();
  });
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });

    elements.video.srcObject = stream;
    await elements.video.play();
    await waitForVideoReady(elements.video);
    resizeCanvas();
    setPillState(elements.cameraState, "Camera live", "good");
  } catch (error) {
    setPillState(elements.cameraState, "Camera blocked", "warn");
    elements.primaryCue.textContent = "Camera access is required.";
    elements.secondaryCue.textContent =
      "Allow webcam permission and reload the page to start the coach.";
    throw error;
  }
}

// Create the pose estimator and point it at the locally bundled MediaPipe assets.
async function createPoseTracker() {
  const pose = new Pose({
    locateFile: (file) => new URL(`../node_modules/@mediapipe/pose/${file}`, import.meta.url).href,
  });

  pose.setOptions({
    modelComplexity: 2,
    smoothLandmarks: true,
    enableSegmentation: false,
    smoothSegmentation: false,
    selfieMode: true,
    minDetectionConfidence: 0.45,
    minTrackingConfidence: 0.45,
  });

  pose.onResults(processResults);
  return pose;
}

// App entry point: start the webcam, start MediaPipe, then feed each animation-frame image
// through the pose model.
async function run() {
  if (!navigator.mediaDevices?.getUserMedia) {
    elements.primaryCue.textContent = "This browser cannot access the webcam API.";
    elements.secondaryCue.textContent = "Use a recent version of Chrome, Edge, or another modern browser.";
    return;
  }

  await startCamera();
  const pose = await createPoseTracker();

  async function frameLoop() {
    await pose.send({ image: elements.video });
    requestAnimationFrame(frameLoop);
  }

  frameLoop();
}

window.addEventListener("resize", resizeCanvas);
run().catch((error) => {
  console.error(error);
});
