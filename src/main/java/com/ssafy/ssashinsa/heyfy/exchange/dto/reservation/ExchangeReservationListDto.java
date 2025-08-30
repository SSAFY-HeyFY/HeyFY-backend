package com.ssafy.ssashinsa.heyfy.exchange.dto.reservation;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
@Schema(description = "환전 예약 내역 응답 dto")
public class ExchangeReservationListDto {
    private List<ExchangeReservationItemDto> data;
}
