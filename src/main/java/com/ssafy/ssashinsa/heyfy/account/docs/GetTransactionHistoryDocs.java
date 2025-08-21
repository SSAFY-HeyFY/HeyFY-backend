package com.ssafy.ssashinsa.heyfy.account.docs;

import com.ssafy.ssashinsa.heyfy.account.dto.InquireTransactionHistoryResponseRecDto;
import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Tag(name = "Account", description = "계좌 관리 API")
@Operation(summary = "계좌 거래 내역 조회", description = "연결된 계좌의 거래 내역을 조회합니다.")
@ApiResponses({
        @ApiResponse(
                responseCode = "200",
                description = "성공적으로 거래 내역을 조회했습니다.",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = InquireTransactionHistoryResponseRecDto.class)
                )
        ),
        @ApiResponse(
                responseCode = "500",
                description = "서버 내부 오류 또는 API 호출 실패",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class)
                )
        )
})
public @interface GetTransactionHistoryDocs {
}