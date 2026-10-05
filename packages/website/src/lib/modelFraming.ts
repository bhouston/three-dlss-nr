/** Fit a bounding sphere in both camera axes, leaving margin around the subject. */
export function modelFitDistance(radius: number, verticalFovDegrees: number, aspect: number): number {
  if (
    ![radius, verticalFovDegrees, aspect].every(Number.isFinite) ||
    !(radius > 0) ||
    !(aspect > 0) ||
    !(verticalFovDegrees > 0 && verticalFovDegrees < 180)
  ) {
    throw new Error('Cannot frame a scene with invalid bounds or camera settings');
  }
  const vertical = (verticalFovDegrees * Math.PI) / 360;
  const horizontal = Math.atan(Math.tan(vertical) * aspect);
  return (radius * 1.12) / Math.sin(Math.min(vertical, horizontal));
}
