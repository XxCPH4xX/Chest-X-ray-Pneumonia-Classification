export const hosted = process.env.NEXT_PUBLIC_HOSTED === "true";
export const uploadLimitMiB = hosted ? 3 : 8;
