package com.ssafy.ssashinsa.heyfy.exchange.dto.exchange;

import com.fasterxml.jackson.annotation.JsonProperty;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
@Schema(description = "환전 응답 dto")
public class ExchangeResponseDto {
    @Schema(description = "입금 계좌 잔액", example = "100000.0")
    private Double depositAccountBalance;
    @Schema(description = "출금 계좌 잔액", example = "500000.0")
    private Double withdrawalAccountBalance;
    @Schema(description = "환전 금액(환전통화기준)", example = "100.0")
    private Double transactionBalance;
    @JsonProperty("isCorrect")
    @Schema(description = "pin 번호 일치 여부", example = "true")
    private boolean isCorrect;
}
