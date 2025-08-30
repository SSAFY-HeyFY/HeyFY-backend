package com.ssafy.ssashinsa.heyfy.exchange.dto.reservation;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Schema(description = "환전 예약 취소 응답 dto")
public class ExchangeReservationCancelResponseDto {
    private boolean success;
}
