package com.ssafy.ssashinsa.heyfy.exchange.dto.reservation;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
@Schema(description = "환전 예약 취소 요청 dto")
public class ExchangeReservationCancelRequestDto {
    @Schema(description = "환전 예약 id", example = "1")
    private Long reservationId;
    @Schema(description = "pin", example = "000000")
    private String pinNumber;
}

