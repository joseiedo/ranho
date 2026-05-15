/// Normalization constants from normalization.json
pub const MAX_AMOUNT: f32 = 10_000.0;
pub const MAX_INSTALLMENTS: f32 = 12.0;
pub const AMOUNT_VS_AVG_RATIO: f32 = 10.0;
pub const MAX_MINUTES: f32 = 1440.0;
pub const MAX_KM: f32 = 1000.0;
pub const MAX_TX_COUNT_24H: f32 = 20.0;
pub const MAX_MERCHANT_AVG_AMOUNT: f32 = 10_000.0;

pub const DEFAULT_MCC_RISK: f32 = 0.5;

/// hour_of_day / 23  (indices 0–23)
pub const HOUR_LUT: [f32; 24] = [
     0.0 / 23.0,  1.0 / 23.0,  2.0 / 23.0,  3.0 / 23.0,
     4.0 / 23.0,  5.0 / 23.0,  6.0 / 23.0,  7.0 / 23.0,
     8.0 / 23.0,  9.0 / 23.0, 10.0 / 23.0, 11.0 / 23.0,
    12.0 / 23.0, 13.0 / 23.0, 14.0 / 23.0, 15.0 / 23.0,
    16.0 / 23.0, 17.0 / 23.0, 18.0 / 23.0, 19.0 / 23.0,
    20.0 / 23.0, 21.0 / 23.0, 22.0 / 23.0, 23.0 / 23.0,
];

/// day_of_week / 6  (Mon=0, Sun=6)
pub const DOW_LUT: [f32; 7] = [
    0.0 / 6.0, 1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0,
    4.0 / 6.0, 5.0 / 6.0, 6.0 / 6.0,
];

#[inline]
pub fn round4(x: f32) -> f32 {
    (x * 10_000.0).round() / 10_000.0
}

/// Clamp to [0.0, 1.0] and round to 4 decimal places.
#[inline]
pub fn clamp01(x: f32) -> f32 {
    round4(x.clamp(0.0, 1.0))
}
